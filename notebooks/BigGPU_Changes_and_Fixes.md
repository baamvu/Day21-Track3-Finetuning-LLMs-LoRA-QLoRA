# Lab 21 — BigGPU (A100/L4) Notebook: Changes & Bug Fixes Log

> **File**: `notebooks/Lab21_LoRA_Finetuning_BigGPU.ipynb`  
> **Base**: `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`  
> **Date**: 2026-06-25

---

## 1. Thay đổi cấu hình so với T4

| Setting | T4 | BigGPU (A100/L4) | Lý do |
|---------|-----|-------------------|-------|
| **Model** | `unsloth/Qwen2.5-3B-bnb-4bit` | `unsloth/Qwen2.5-7B-bnb-4bit` | 40-80GB VRAM dư sức chạy 7B |
| **Train batch size** | 1 | 4 | A100 có bandwidth + VRAM lớn hơn |
| **Gradient accumulation** | 8 | 2 | Effective batch vẫn = 8 |
| **Eval strategy** | `"no"` (tắt) | `"steps"`, eval_steps=50 | A100 đủ VRAM cho mid-train eval |
| **Packing** | `False` | `False` | Unsloth packing bug với Qwen2.5 GQA (xem Bug #1) |
| **Max seq length cap** | 1024 | 2048 | A100 handle được sequences dài hơn |
| **Dataset samples** | 200 | 500 | Nhiều data hơn → kết quả experiment tin cậy hơn |
| **Qualitative prompts** | 5 | 10 | Đa dạng đánh giá hơn |
| **GPU cost** | $0.35/hr (T4 free) | $1.10/hr (Colab Pro+) | Giá A100 trên Colab |
| **Output dir** | `/content/lab21_lora_t4` | `/content/lab21_lora_biggpu` | Tránh ghi đè nếu chạy cả 2 |

---

## 2. Tính năng mới

### 2.1 Eval during training

- **T4**: `eval_strategy = "no"` — chỉ có train loss, không detect overfitting
- **BigGPU**: `eval_strategy = "steps"`, `eval_steps = 50`
- **Lợi ích**: Loss curve có cả train + eval, thấy overfitting real-time
- **Ảnh hưởng thời gian**: Thêm ~30-45 giây mỗi rank (3 lần eval × 10-15s)

### 2.2 Base model perplexity (Section 4b)

- **T4**: Không tính — thiếu row "Base" trong bảng so sánh
- **BigGPU**: Tính perplexity base model (chưa fine-tune) bằng manual eval loop
- **Lợi ích**: Đủ 4 rows theo rubric (Base + r=8 + r=16 + r=64), tính được % cải thiện
- **Lưu ý**: Không dùng `SFTTrainer` cho base model (xem Bug #2)

### 2.3 Loss curve PNG export

- **T4**: Chỉ `plt.show()` — mất khi runtime disconnect
- **BigGPU**: `plt.savefig(OUTPUT_DIR + "loss_curve_r{rank}.png", dpi=150)` + show
- **Files export**: `loss_curve_r16.png`, `loss_curve_r8.png`, `loss_curve_r64.png`

### 2.4 Token length distribution PNG

- **T4**: Chỉ show, không save
- **BigGPU**: Save `token_length_distribution.png` vào OUTPUT_DIR
- **Lợi ích**: Nhúng trực tiếp vào REPORT.md, giảng viên verify được

### 2.5 GPU capability detection

- **T4**: Chỉ check VRAM > 20GB
- **BigGPU**: Check thêm compute capability (`cc[0] >= 8` → Ampere+), BF16 support
- **Lợi ích**: Cảnh báo đúng nếu học viên lỡ mở notebook sai GPU

---

## 3. Bugs gặp phải & Fixes

### Bug #1: Unsloth packing + Qwen2.5 GQA tensor mismatch

**Lỗi:**
```
RuntimeError: The size of tensor a (520) must match the size of tensor b (1413)
at non-singleton dimension 3
```

**Nguyên nhân:**
- Unsloth log: `"Hugging Face's packing is currently buggy - we're disabling it for now!"`
- Nhưng ngay sau đó: `"Packing enabled - training is >2x faster and uses less VRAM!"`
- Unsloth tắt packing của HuggingFace nhưng **tự bật packing riêng**
- Packing gộp nhiều samples thành 1 sequence → Q length ≠ KV length
- Qwen2.5 dùng GQA (Grouped Query Attention) → scaled_dot_product_attention crash

**Fix:**
```python
# Tất cả 3 chỗ trong make_trainer:
packing=False  # thay vì True
```

**Ảnh hưởng:** Training chậm hơn ~20-30% so với packing, nhưng chạy ổn định.

---

### Bug #2: SFTTrainer từ chối quantized model không có adapter

**Lỗi:**
```
ValueError: You cannot perform fine-tuning on purely quantized models.
Please attach trainable adapters on top of the quantized model...
```

**Nguyên nhân:**
- `transformers ≥ 5.x` thêm validation: model 4-bit mà không có PEFT adapter → không cho tạo Trainer
- Cell 4b dùng `SFTTrainer` để tính base model perplexity, nhưng base model chưa wrap LoRA

**Fix:** Bỏ `SFTTrainer`, dùng manual eval loop với `DataLoader` + `torch.tensor()`:
```python
# Không dùng SFTTrainer cho base model
# Dùng DataLoader thủ công:
for i in range(0, n_samples, batch_size):
    batch_rows = eval_tokenized[i:i+batch_size]
    input_ids = torch.tensor(batch_rows["input_ids"]).to("cuda")
    attention_mask = torch.tensor(batch_rows["attention_mask"]).to("cuda")
    outputs = base_for_ppl(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
```

---

### Bug #3: `datasets` + `torchvision` VideoReader import error

**Lỗi:**
```
ImportError: cannot import name 'VideoReader' from 'torchvision.io'
```

**Nguyên nhân:**
- `datasets` library gọi `set_format(type="torch")` → trigger `torchvision.io.VideoReader` import
- Version `torchvision` trên Colab không có `VideoReader` (compatibility issue)

**Fix:** Không dùng `set_format("torch")` + không dùng `DataLoader`. Convert sang tensor thủ công:
```python
# TRƯỚC (gây lỗi):
eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_loader = DataLoader(eval_tokenized, batch_size=4)
for batch in eval_loader:  # ← crash

# SAU (không lỗi):
# Không set_format, loop bằng index
for i in range(0, n_samples, batch_size):
    batch_rows = eval_tokenized[i:i+batch_size]
    input_ids = torch.tensor(batch_rows["input_ids"]).to("cuda")
```

---

### Warning: `warmup_ratio` deprecated

**Cảnh báo:**
```
warmup_ratio is deprecated and will be removed in v5.2.
Use `warmup_steps` instead.
```

**Fix:**
```python
# TRƯỚC:
warmup_ratio=0.10,

# SAU:
warmup_steps=17,  # ~10% của 171 total steps
```

---

### Warning: `use_return_dict` deprecated

**Cảnh báo:**
```
`use_return_dict` is deprecated! Use `return_dict` instead!
```

**Fix:** Không cần sửa — warning từ Unsloth internals, không ảnh hưởng training.

---

### Warning: PAD/BOS/EOS token mismatch

**Cảnh báo:**
```
Unsloth: pad_token was a vision token (<|vision_pad|>) on a text-only model.
Replaced with hôtel to avoid NaN losses.
unsloth/Qwen2.5-7B-bnb-4bit does not have a padding token!
Will use pad_token = <|PAD_TOKEN|>.
```

**Fix:** Không cần sửa — Unsloth tự xử lý. Warning này bình thường với Qwen2.5.

---

## 4. Thời gian ước tính

| Phần | A100 40GB | Ghi chú |
|------|-----------|---------|
| Setup + install | ~2-3 phút | pip install, download model |
| Dataset prep | ~1-2 phút | 500 samples, tokenize |
| Train r=16 | ~8-10 phút | ~171 steps + 3 lần eval |
| Train r=8 | ~6-8 phút | Ít params hơn |
| Train r=64 | ~12-15 phút | Nhiều params nhất |
| Base perplexity | ~30 giây | Manual eval loop |
| Eval 3 ranks | ~2-3 phút | safe_evaluate() |
| Qualitative (10 prompts) | ~3-5 phút | Generate 20 responses |
| **Tổng** | **~35-50 phút** | |

So với T4 (~60 phút cho model 3B), A100 train model 7B lớn hơn 2.3× nhưng **vẫn nhanh hơn** nhờ BF16 native + FlashAttention + batch size 4.

---

## 5. Checklist trước khi chạy

- [ ] Chọn runtime: **A100 40GB** hoặc **L4 24GB** (không dùng T4)
- [ ] `packing=False` trong tất cả 3 chỗ của `make_trainer`
- [ ] `warmup_steps=17` thay vì `warmup_ratio=0.10`
- [ ] Cell 4b dùng manual eval loop (không dùng SFTTrainer cho base model)
- [ ] Không dùng `set_format("torch")` (tránh VideoReader error)
- [ ] Kiểm tra OUTPUT_DIR có đủ files sau khi chạy xong

---

## 6. Files export sau khi chạy xong

```
/lab21_lora_biggpu/
├── r8/                              ← adapter checkpoint (r=8)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── r16/                             ← adapter checkpoint (r=16)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── r64/                             ← adapter checkpoint (r=64)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── rank_experiment_summary.csv      ← bảng metrics 4 rows
├── qualitative_comparison.csv       ← 10 before/after examples
├── loss_curve_r16.png              ← training loss + eval loss curve
├── loss_curve_r8.png               ← training loss + eval loss curve
├── loss_curve_r64.png              ← training loss + eval loss curve
└── token_length_distribution.png   ← histogram token lengths
```
