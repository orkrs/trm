from __future__ import annotations

import gc
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from config import CONFIG, TrainingConfig
from memory.hierarchical_memory import HierarchicalMemory

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None  # type: ignore[assignment,misc]

try:
    from bitsandbytes.optim import PagedAdamW
except ImportError:
    PagedAdamW = None  # fallback defined in __init__


class TruncatedBPTTDataset(IterableDataset):
    """Wrapper that splits long sequences into truncated chunks.

    Each chunk has length at most truncation_length.  Chunks from
    the same original sequence are yielded sequentially so the
    training loop can carry hidden state across chunks and detach
    at segment boundaries.

    Supports DDP sharding: when ``world_size > 1``, each rank sees
    a disjoint subset of sequences (round-robin).  This ensures
    each GPU processes different data during distributed training.

    Args:
        data: List of tokenized sequences (variable length).
        truncation_length: Maximum tokens per chunk.
        seq_length: Maximum total sequence length (sequences longer
                    than this are split into separate items).
        rank: DDP rank (0-indexed).  ``0`` means no sharding.
        world_size: Total number of DDP processes.  ``1`` means no sharding.
    """

    def __init__(
        self,
        data: List[torch.Tensor],
        truncation_length: int = 2048,
        seq_length: int = 8192,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.data: List[torch.Tensor] = data
        self.truncation_length: int = truncation_length
        self.seq_length: int = seq_length
        self.rank: int = rank
        self.world_size: int = world_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        data = self.data

        # DDP sharding: each rank gets a disjoint subset.
        if self.world_size > 1:
            data = data[self.rank::self.world_size]

        # DataLoader worker sharding: each worker gets a subset of the rank's data.
        if worker_info is not None:
            data = data[worker_info.id::worker_info.num_workers]

        for seq in data:
            seq = seq[:self.seq_length]
            for start in range(0, len(seq), self.truncation_length):
                end = min(start + self.truncation_length, len(seq))
                chunk = seq[start:end]
                if chunk.numel() < 2:
                    continue
                yield {
                    "input_ids": chunk.unsqueeze(0),
                    "labels": chunk.unsqueeze(0),
                    "segment_start": start,
                    "is_last": end >= len(seq),
                }

    def __len__(self) -> int:
        total = 0
        data = self.data
        if self.world_size > 1:
            data = data[self.rank::self.world_size]
        for seq in data:
            total += max(1, (min(len(seq), self.seq_length) + self.truncation_length - 1)
                         // self.truncation_length)
        return total


class TRMBankTrainer:
    """Truncated BPTT trainer for TRM-Bank v3.0.

    Trains only the QRandLoRA adaptation parameters and the LPRM
    head while keeping the pretrained Mamba backbone frozen.

    The training loop:
        1. Splits long sequences into truncation_length segments.
        2. Processes each segment sequentially, detaching the hidden
           state at segment boundaries (truncated BPTT).
        3. Accumulates gradients across segments within a sequence.
        4. Applies gradient clipping and optimizer step per sequence.

    Args:
        model: The TRMBankModel instance (must be built).
        lprm: Optional MultiHeadLPRM for quality prediction (trained jointly).
        config: Training configuration.
        train_dataset: IterableDataset yielding truncated chunks.
        eval_dataset: Optional dataset for evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        lprm: Optional[nn.Module] = None,
        config: TrainingConfig = CONFIG.training,
        train_dataset: Optional[IterableDataset] = None,
        eval_dataset: Optional[IterableDataset] = None,
        memory: Optional[HierarchicalMemory] = None,
        accelerator: Optional[Any] = None,
    ) -> None:
        self.model: nn.Module = model
        self.lprm: Optional[nn.Module] = lprm
        self.config: TrainingConfig = config
        self.train_dataset: Optional[IterableDataset] = train_dataset
        self.eval_dataset: Optional[IterableDataset] = eval_dataset
        self.memory: Optional[HierarchicalMemory] = memory
        self.accelerator: Optional[Any] = accelerator

        # Collect trainable parameters (QRandLoRA + LPRM only).
        trainable_params: List[torch.Tensor] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                if self._is_main_process:
                    logger.debug(f"Trainable param: {name}, shape={param.shape}")

        if self.lprm is not None:
            trainable_params.extend(
                p for p in self.lprm.parameters() if p.requires_grad
            )

        if not trainable_params:
            if self._is_main_process:
                logger.warning("No trainable parameters found. Check requires_grad flags.")

        # With DDP (accelerate), avoid PagedAdamW (it has device_map issues).
        # With device_map="auto" (single-process), PagedAdamW is OK on 1 GPU.
        has_ddp = self.accelerator is not None and torch.cuda.device_count() > 1
        use_paged: bool = (
            PagedAdamW is not None
            and torch.cuda.is_available()
            and not has_ddp
            and torch.cuda.device_count() == 1
        )
        optim_cls = PagedAdamW if use_paged else torch.optim.AdamW
        self.optimizer = optim_cls(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if self._is_main_process:
            if use_paged:
                logger.info("Using PagedAdamW optimizer (CPU-offloaded Adam states).")
            else:
                logger.info("Using standard AdamW optimizer.")

        # Linear warmup + cosine decay scheduler.
        total_steps = config.num_epochs * (
            len(train_dataset) if train_dataset is not None else 1000
        )
        warmup_steps = config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # Wrap model, optimizer, scheduler with accelerate for DDP.
        if self.accelerator is not None:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )

        self.global_step: int = 0
        self.best_eval_loss: float = float("inf")

    @property
    def _is_main_process(self) -> bool:
        """Return True on the main process (or always if no accelerator)."""
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy language modeling loss.

        Args:
            logits: Predicted logits of shape (B, L, V).
            labels: Target token ids of shape (B, L).

        Returns:
            Scalar loss tensor.
        """
        B, L, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def _compute_lprm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPRM auxiliary loss from model confidence.

        For each token position, the target reward is the model's
        predicted probability of the correct token (soft target in
        [0, 1]).  All LPRM heads predict this same target, teaching
        them to recognise when the model is confident vs uncertain.

        Args:
            logits: Predicted logits of shape (B, L, V).
            labels: Target token ids of shape (B, L).
            hidden_states: Final layer hidden states of shape (B, L, D).

        Returns:
            Scalar LPRM loss tensor.
        """
        if self.lprm is None:
            return torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=-1)                              # (B, L, V)
        p_correct = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
        target = p_correct.unsqueeze(-1)                                 # (B, L, 1)

        lprm_device = next(self.lprm.parameters()).device
        q_pred = self.lprm(hidden_states.to(lprm_device))               # (B, L, M)
        if target.device != q_pred.device:
            target = target.to(q_pred.device)
        loss = F.mse_loss(q_pred, target.expand_as(q_pred))
        return loss

    def train_lprm_on_replay(
        self,
        batch_size: int = 16,
        lr: float = 1e-4,
        num_steps: int = 32,
    ) -> float:
        """Train LPRM heads from the memory replay buffer.

        Samples experiences generated during inference and trains the
        LPRM head to predict the actual reward from the hidden state
        that was used at auction time.

        Args:
            batch_size: Number of experiences per step.
            lr: Learning rate for the inner optimiser.
            num_steps: Number of gradient steps.

        Returns:
            Average LPRM replay loss over the last batch.
        """
        if self.lprm is None or self.memory is None:
            if self._is_main_process:
                logger.warning("No LPRM or memory attached; skipping replay training.")
            return 0.0

        replay = self.memory.replay
        if len(replay) < 2:
            if self._is_main_process:
                logger.warning(f"Replay buffer too small ({len(replay)}); skipping.")
            return 0.0

        self.lprm.train()
        optim = torch.optim.AdamW(self.lprm.parameters(), lr=lr)

        # With device_map="auto", LPRM may be on a different GPU than the
        # first model parameter.  Use the LPRM's own device.
        lprm_device = next(self.lprm.parameters()).device
        module_names = self.lprm.module_names if hasattr(self.lprm, "module_names") else []
        name_to_idx = {name: i for i, name in enumerate(module_names)}

        avg_loss: float = 0.0
        for step in range(num_steps):
            exps = replay.sample(batch_size)
            if not exps:
                break

            hidden_states: List[torch.Tensor] = []
            targets: List[float] = []
            for exp in exps:
                hidden_states.append(exp.hidden_for_lprm.to(lprm_device))
                targets.append(exp.reward)

            h: torch.Tensor = torch.cat(hidden_states, dim=0)           # (B, D)

            module_idx = name_to_idx.get(exps[0].module_name, 0)
            q_pred: torch.Tensor = self.lprm(h, module_idx=module_idx)  # (B, 1)
            true = torch.tensor(targets, device=lprm_device, dtype=q_pred.dtype).unsqueeze(-1)
            # true: (B, 1)

            loss = F.mse_loss(q_pred, true)
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss = loss.item()

        self.lprm.eval()
        return avg_loss

    def train_epoch(self, epoch: int,
                    max_steps_remaining: int = -1) -> Dict[str, float]:
        """Run one training epoch with truncated BPTT.

        States are carried across segments of the same sequence
        and detached at segment boundaries.

        Args:
            epoch: Current epoch number (for logging).
            max_steps_remaining: If > 0, stop after this many optimizer
                                 steps.  -1 means no limit.

        Returns:
            Dictionary of average metrics for the epoch.
        """
        if self.train_dataset is None:
            raise RuntimeError("No training dataset provided.")

        self.model.train()
        if self.lprm is not None:
            self.lprm.train()

        total_loss: float = 0.0
        total_lprm_loss: float = 0.0
        total_tokens: int = 0
        num_batches: int = 0
        states: Any = None

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=0,
        )

        # Wrap dataloader with accelerate for DDP sharding.
        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="chunk",
                    disable=not self._is_main_process)
        amp_enabled: bool = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        for batch in pbar:
            input_ids = batch["input_ids"]             # (1, L_chunk)
            labels = batch["labels"]                   # (1, L_chunk)
            is_last = batch["is_last"]

            # With accelerate, tensors are already on the correct device.
            # Without accelerate, move to model's first parameter device.
            if self.accelerator is None:
                first_device = next(self.model.parameters()).device
                input_ids = input_ids.to(first_device)
                labels = labels.to(first_device)

            kwargs: Dict[str, Any] = {"input_ids": input_ids}
            if states is not None:
                kwargs["states"] = states

            # Try return_states; some models (e.g. TinyTRMModel) support it.
            # If the model doesn't accept it, catch TypeError.
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                try:
                    outputs = self.model(return_states=True, **kwargs)
                    states = outputs.get("states", None)
                except TypeError:
                    outputs = self.model(**kwargs)

                logits = outputs["logits"]
                hidden_states = outputs.get("hidden_states", None)

                # Labels must be on the same device as logits
                # (with device_map="auto", logits may be on the last GPU).
                if labels.device != logits.device:
                    labels = labels.to(logits.device)

                # Language modelling loss.
                lm_loss = self._compute_loss(logits, labels)
                loss = lm_loss / self.config.gradient_accumulation_steps

                # LPRM auxiliary loss (teaches heads to predict model confidence).
                if self.lprm is not None and hidden_states is not None:
                    # hidden_states is a tuple of tensors (one per layer).
                    # Use the last layer's hidden states for LPRM.
                    if isinstance(hidden_states, (tuple, list)):
                        hidden_states = hidden_states[-1]
                    # Move hidden_states to logits device for loss consistency.
                    # With device_map="auto", logits may be on cuda:0 (from lm_head)
                    # while hidden_states is on cuda:1 (from last layer).
                    if hidden_states.device != logits.device:
                        hidden_states = hidden_states.to(logits.device)
                    lprm_loss = self._compute_lprm_loss(logits, labels, hidden_states)
                    # lprm_loss may be on a different device than loss (from LPRM on cuda:1).
                    if lprm_loss.device != loss.device:
                        lprm_loss = lprm_loss.to(loss.device)
                    loss = loss + (self.config.lprm_weight * lprm_loss)
                    total_lprm_loss += lprm_loss.item()

            # Backward pass: use accelerator if available, else plain backward.
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            total_loss += lm_loss.item()
            total_tokens += labels.numel()
            num_batches += 1

            # Gradient accumulation step.
            if (num_batches % self.config.gradient_accumulation_steps == 0) or is_last:
                if self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Early stopping at max_steps.
                if max_steps_remaining > 0 and self.global_step >= max_steps_remaining:
                    if self._is_main_process:
                        logger.info(f"Reached max_steps={max_steps_remaining}, stopping early.")
                    break

                if self.global_step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            # Logging (main process only).
            if self._is_main_process and self.global_step % self.config.log_every_n_steps == 0 and num_batches > 0:
                avg_loss = total_loss / max(1, num_batches)
                avg_lprm = total_lprm_loss / max(1, num_batches)
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {self.global_step} | loss={avg_loss:.4f} | "
                    f"lprm={avg_lprm:.4f} | lr={lr:.2e} | tokens={total_tokens}"
                )
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lprm=f"{avg_lprm:.4f}", lr=f"{lr:.2e}")

            # Evaluation.
            if (self.global_step % self.config.eval_every_n_steps == 0
                    and self.eval_dataset is not None):
                eval_metrics = self.evaluate()
                if self._is_main_process:
                    logger.info(f"Eval at step {self.global_step}: {eval_metrics}")
                if eval_metrics.get("loss", float("inf")) < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["loss"]
                    self._save_checkpoint("best")
                self.model.train()

            # Detach states for truncated BPTT at segment boundaries.
            if is_last:
                states = None
            elif states is not None:
                states = [s.detach() for s in states]

        avg_epoch_loss = total_loss / max(1, num_batches)
        avg_epoch_lprm = total_lprm_loss / max(1, num_batches)
        if self._is_main_process:
            logger.info(
                f"Epoch {epoch} complete. avg_loss={avg_epoch_loss:.4f} "
                f"avg_lprm_loss={avg_epoch_lprm:.4f}"
            )
        return {"loss": avg_epoch_loss, "lprm_loss": avg_epoch_lprm, "tokens": total_tokens}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the evaluation dataset.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {"loss": float("nan")}

        self.model.eval()
        if self.lprm is not None:
            self.lprm.eval()

        total_loss: float = 0.0
        num_batches: int = 0

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=None,
            num_workers=0,
        )

        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            # With DDP / device_map, tensors are already on the correct device.
            # Fallback: move to first parameter's device.
            if self.accelerator is None:
                first_device = next(self.model.parameters()).device
                input_ids = input_ids.to(first_device)
                labels = labels.to(first_device)

            outputs = self.model(input_ids=input_ids)
            logits = outputs["logits"]
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            loss = self._compute_loss(logits, labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return {"loss": avg_loss}

    def _save_checkpoint(self, tag: str = "latest") -> str:
        """Save model and optimizer state.

        Uses ``accelerator.save_state`` when an accelerator is attached,
        otherwise falls back to manual ``torch.save``.

        Args:
            tag: Checkpoint tag (e.g., "latest", "best").

        Returns:
            Path to the saved checkpoint.
        """
        path = f"{self.config.output_dir}/checkpoint_{tag}"

        if self.accelerator is not None:
            self.accelerator.save_state(path)
            if self._is_main_process:
                logger.info(f"Checkpoint saved via accelerate to {path}")
        else:
            path = path + ".pt"
            state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
            }
            if self.lprm is not None:
                state["lprm_state"] = self.lprm.state_dict()
            torch.save(state, path)
            logger.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint.

        Uses ``accelerator.load_state`` when an accelerator is attached,
        otherwise falls back to manual ``torch.load``.

        Args:
            path: Path to the checkpoint file or directory.
        """
        if self.accelerator is not None:
            self.accelerator.load_state(path)
            if self._is_main_process:
                logger.info(f"Checkpoint loaded via accelerate from {path}")
        else:
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state["model_state"], strict=False)
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.scheduler.load_state_dict(state["scheduler_state"])
            self.global_step = state["global_step"]
            self.best_eval_loss = state["best_eval_loss"]
            if self.lprm is not None and "lprm_state" in state:
                self.lprm.load_state_dict(state["lprm_state"])
            logger.info(f"Checkpoint loaded from {path} (step {self.global_step})")

    def train(self, num_epochs: Optional[int] = None,
              max_steps: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Override for the configured number of epochs.
            max_steps: Override for the configured max_steps.  When set,
                       training stops after this many optimizer steps
                       (across all epochs).

        Returns:
            Dictionary of training history.
        """
        n_epochs = num_epochs or self.config.num_epochs
        total_max = max_steps if max_steps is not None else self.config.max_steps
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_lprm_loss": [],
            "eval_loss": [],
        }

        for epoch in range(1, n_epochs + 1):
            remaining = -1
            if total_max > 0:
                remaining = max(0, total_max - self.global_step)
                if remaining <= 0:
                    if self._is_main_process:
                        logger.info(f"max_steps={total_max} already reached.")
                    break

            train_metrics = self.train_epoch(epoch,
                                             max_steps_remaining=remaining)
            history["train_loss"].append(train_metrics["loss"])
            history["train_lprm_loss"].append(train_metrics.get("lprm_loss", 0.0))

            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                history["eval_loss"].append(eval_metrics["loss"])
                self._save_checkpoint(f"epoch_{epoch}")

            if total_max > 0 and self.global_step >= total_max:
                break

        self._save_checkpoint("final")
        return history
