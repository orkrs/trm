# TRM-Bank v3.0: CODE_REALITY

Честный разбор: что написано в коде vs что обещано в ARTICLE.md.

---

## 1. Загрузка Mamba-2.8B (4-bit NF4)

**ARTICLE.md:** Загружаем state-spaces/mamba-1.4b-hf через bitsandbytes в 4-bit NF4.

**Код:** `core/complex_mimo_mamba.py:511-556`

```python
# Работает, но требует mamba_ssm + bitsandbytes
# В Colab не встаёт — CUDA mismatch
model = transformers.AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-1.4b-hf",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
)
for param in model.parameters():  # заморозка
    param.requires_grad = False
```

**Реальность:** Код есть, но на практике (Colab, домашний ПК без mamba_ssm) не запускается. Библиотека mamba_ssm требует специфической версии CUDA и кастомных CUDA-ядер.

---

## 2. Замена SSM-скана на ComplexMIMOScan

**ARTICLE.md:** "Заменяем её оригинальный вещественный одноканальный блок Mamba-2 на наш кастомный комплекснозначный многоканальный блок Mamba-3 MIMO."

**Код:** `core/complex_mimo_mamba.py:593-643`

```python
# НЕ замена, а ДОБАВЛЕНИЕ поверх
outputs = self.backbone(input_ids=input_ids, output_hidden_states=True)
for i, hs in enumerate(outputs.hidden_states):
    if i < len(self.custom_scans):
        scan_out, _ = self.custom_scans[i](hs)       # ComplexMIMOScan
        enriched = hs + scan_out.mean(dim=-1)         # residual поверх backbone
```

**Реальность:** Оригинальный Mamba-2 scan не удаляется. ComplexMIMOScan работает параллельно и складывается поверх hidden_state через residual. Это не замена, а enrichment.

`generate()` вообще игнорирует custom_scans — просто делегирует `backbone.generate()` (`core/complex_mimo_mamba.py:677-681`).

---

## 3. QRandLoRA на проекционные слои

**ARTICLE.md:** "Вешаем на её проекционные слои адаптеры RandLoRA... Обучаться будут исключительно эти лёгкие адаптеры (~40MB)."

**Код:** `core/qrrandlora.py:12-285`

```python
# QRandLoRA существует как отдельный модуль:
# Delta_W = sum_j B_j @ Lambda_j @ A_j @ Gamma_j
# A_j, B_j — замёрзшие разреженные тернарные матрицы {-1,0,1}
# Lambda_j, Gamma_j — обучаемые диагональные скейлы
```

**Реальность:** QRandLoRALayer и QRandLoRALinear полностью реализованы и протестированы. **Но они никуда не подключены.** Нет кода, который вешает QRandLoRA на проекции Mamba-2.8B или TinyTRMModel. Это standalone-компонент без интеграции.

---

## 4. TinyTRMModel (запасной вариант)

**ARTICLE.md:** Не упоминается.

**Код:** `core/complex_mimo_mamba.py:319-469`

```python
# Self-contained модель без внешних зависимостей
model = TinyTRMModel(vocab_size=50257, d_model=256, d_state=64,
                     mimo_rank=4, num_layers=2)
# embed → TinyTRMBlock×N → norm → lm_head
# Не требует mamba_ssm, bitsandbytes, transformers
# Работает на CPU, в Colab, где угодно
```

**Реальность:** TinyTRMModel — компромисс, написанный чтобы проект хоть как-то работал без Mamba-2.8B. У него 2 слоя и d_model=256 (против 64 слоёв и d_model=2560 у оригинала). Это proof-of-concept, а не замена.

---

## 5. Pipeline: что реально можно запустить

### Pipeline A: TinyTRMModel (работает везде)

```
input_ids → embed → [TinyTRMBlock × 2] → norm → lm_head → logits
                          │
                    ComplexMIMOScan
                    (d_state=64, mimo_rank=4)
                          │
                    R ветвей → mean → residual + LayerNorm
```

```python
from core import TinyTRMModel

model = TinyTRMModel(vocab_size=100, d_model=32, d_state=16, mimo_rank=2)

# forward
out = model(torch.randint(0, 100, (2, 8)))
logits = out["logits"]          # (2, 8, 100)

# генерация с переносом состояний
gen = model.generate(inputs, max_new_tokens=32, temperature=0.0)

# обучение с Truncated BPTT
from training import TRMBankTrainer, TruncatedBPTTDataset
data = [torch.randint(0, 100, (50,))]
ds = TruncatedBPTTDataset(data, truncation_length=20)
trainer = TRMBankTrainer(model=model, train_dataset=ds)
trainer.train_epoch(1)
```

### Pipeline B: TRMBankModel (требует mamba_ssm)

```
input_ids → Mamba-2.8B backbone (4-bit frozen)
                ↓
           hidden_states
                ↓
     [ComplexMIMOScan × 64] → average R branches → residual
                ↓
           enriched logits
```

```python
from core import TRMBankModel

model = TRMBankModel(pretrained_name="state-spaces/mamba-1.4b-hf",
                     mimo_rank=4, d_state=64)
model.build()  # загружает Mamba-2.8B, замораживает, создаёт 64 скана

# forward — backbone + enrichment
out = model(torch.randint(0, 50257, (1, 128)))

# generate — чистый backbone (custom_scans НЕ участвуют)
gen = model.generate(inputs, max_new_tokens=256)
```

---

## 6. Что отвалилось / недоделано

| Компонент | Статус | Проблема |
|---|---|---|
| Mamba-2.8B 4-bit загрузка | ✅ Реализовано | Не работает без mamba_ssm на Colab |
| Замена скана | ⚠️ Частично | Не замена, а enrichment; generate() игнорирует |
| QRandLoRA на проекции | ❌ Не подключено | Код есть, но не встроен в модель |
| Trapezoidal trap-параметр | ⚠️ Не используется | Вычисляется в in_proj, но нигде не применяется |
| Иерархическая память | ✅ Реализовано | Работает, но не привязана к генерации |
| Теоретико-игровой роутер | ✅ Реализовано | Работает, но не интегрирован с forward |
| TinyTRMModel | ✅ Реализовано | Работает, но игрушечный (2 слоя, 256 dim) |
| LPRM оценка качества | ✅ Реализовано | Работает, heads не обучены |
| SymPy/OR-Tools sandbox | ✅ Реализовано | Работают как отдельные утилиты |

---

## 7. Диаграмма зависимостей (что от чего зависит)

```
TinyTRMModel ───────> ComplexMIMOScan ───> _rotate_2d_pairs
                                              einops, torch
TRMBankModel ───────> ComplexMIMOScan
                  ───> Mamba-2.8B (mamba_ssm + bitsandbytes)
QRandLoRALayer ─────> torch (никак не связан с моделями)
GameTheoreticRouter ─> LPRM, ComputeProvider[5 шт]
HierarchicalMemory ──> FAISSIndex, SemanticPredictor
TRMBankTrainer ──────> TinyTRMModel | TRMBankModel
                       TruncatedBPTTDataset
```
