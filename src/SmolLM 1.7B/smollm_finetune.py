"""
SmolLM2-1.7B-Instruct · Full Fine-Tuning on Recast-30k
========================================================
Target environment : Google Colab Pro  (A100 / L4 40GB recommended)
Model              : HuggingFaceTB/SmolLM2-1.7B-Instruct
Task               : Constraint-following instruction tuning
"""

# ── 0. INSTALL (run this cell first in Colab) ────────────────────────────────
# !pip install -q transformers==4.44.2 datasets accelerate bitsandbytes \
#              trl==0.10.1 peft evaluate rouge_score bert_score wandb

import os, json, math, time, random
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer, SFTConfig
import wandb

# ── 1. CONFIG ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_id        : str   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # Data
    data_path       : str   = "RECAST-30K.json"   # local file, or HF path
    train_ratio     : float = 0.85
    val_ratio       : float = 0.10
    test_ratio      : float = 0.05
    max_samples     : int   = 30_000              # cap if you want a quick run
    max_seq_len     : int   = 1024
    # Training
    output_dir      : str   = "./smollm2_finetuned"
    num_epochs      : int   = 3
    per_device_bs   : int   = 4                   # A100: try 8; L4: 4
    grad_accum      : int   = 4                   # effective batch = 16
    lr              : float = 2e-5
    lr_scheduler    : str   = "cosine"
    warmup_ratio    : float = 0.05
    weight_decay    : float = 0.01
    max_grad_norm   : float = 1.0
    # Precision
    bf16            : bool  = True                # A100 supports bf16; else fp16
    fp16            : bool  = False
    # Logging
    logging_steps   : int   = 10
    eval_steps      : int   = 200
    save_steps      : int   = 200
    save_total_limit: int   = 2
    load_best_model : bool  = True
    metric_for_best : str   = "eval_loss"
    use_wandb       : bool  = True
    wandb_project   : str   = "recast30k-smollm2-finetune"
    seed            : int   = 42

cfg = Config()

# ── 2. SEED ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# ── 3. WANDB (optional) ──────────────────────────────────────────────────────

if cfg.use_wandb:
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

# ── 4. LOAD & INSPECT DATASET ────────────────────────────────────────────────

print("Loading Recast-30k …")

def load_recast(path: str) -> list[dict]:
    """
    Recast-30k schema (typical):
      {
        "id":          str,
        "instruction": str,          # the task description
        "constraints": list[str],    # list of constraint strings
        "context":     str | null,   # optional background
        "response":    str           # gold output
      }
    Adjust field names below if your version differs.
    """
    with open(path) as f:
        data = json.load(f)
    # If it's a dict with a key like "data" or "samples", unwrap:
    if isinstance(data, dict):
        data = data.get("data", data.get("samples", list(data.values())[0]))
    return data

raw = load_recast(cfg.data_path)
print(f"  Total rows: {len(raw)}")

# Quick schema sniff
sample = raw[0]
print("  Keys found:", list(sample.keys()))
print("  Sample:\n ", json.dumps(sample, indent=2)[:600])

# ── 5. PROMPT TEMPLATE ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise assistant. Read the instruction and satisfy EVERY constraint listed. "
    "Do not skip or partially satisfy any constraint."
)

def format_constraints(constraints) -> str:
    if not constraints:
        return ""
    if isinstance(constraints, str):
        return f"\n\nConstraints:\n- {constraints}"
    return "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

def build_chat_prompt(row: dict, tokenizer, include_response: bool = True) -> str:
    """
    SmolLM2-Instruct uses the ChatML template:
      <|im_start|>system\n…<|im_end|>
      <|im_start|>user\n…<|im_end|>
      <|im_start|>assistant\n…<|im_end|>
    """
    instruction  = row.get("instruction", row.get("prompt", ""))
    constraints  = row.get("constraints", [])
    context      = row.get("context", "") or ""
    response     = row.get("response",    row.get("output", ""))

    user_content = instruction
    if context:
        user_content = f"Context:\n{context}\n\n{user_content}"
    user_content += format_constraints(constraints)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]
    if include_response:
        messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not include_response,
    )

# ── 6. TOKENIZER ─────────────────────────────────────────────────────────────

print("\nLoading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"   # needed for SFTTrainer

# ── 7. TOKENIZE & SPLIT ──────────────────────────────────────────────────────

print("Formatting & tokenizing …")

if len(raw) > cfg.max_samples:
    random.shuffle(raw)
    raw = raw[:cfg.max_samples]

formatted = [{"text": build_chat_prompt(r, tokenizer)} for r in raw]

# Token-length filtering
def token_len(s):
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])

formatted = [x for x in formatted if token_len(x["text"]) <= cfg.max_seq_len]
print(f"  After length filter: {len(formatted)} samples")

# Train / val / test split
random.shuffle(formatted)
n       = len(formatted)
n_train = int(n * cfg.train_ratio)
n_val   = int(n * cfg.val_ratio)

train_data = formatted[:n_train]
val_data   = formatted[n_train : n_train + n_val]
test_data  = formatted[n_train + n_val :]

print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "val":   Dataset.from_list(val_data),
    "test":  Dataset.from_list(test_data),
})

# Save test split for offline evaluation later
test_path = Path(cfg.output_dir) / "test_split.json"
test_path.parent.mkdir(parents=True, exist_ok=True)
with open(test_path, "w") as f:
    json.dump(test_data, f, indent=2)
print(f"  Test split saved → {test_path}")

# ── 8. MODEL ─────────────────────────────────────────────────────────────────

print("\nLoading model …")
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    device_map="auto",           # fills available GPUs
    use_cache=False,             # required when gradient checkpointing is on
)
model.config.use_cache = False
model.enable_input_require_grads()   # needed for gradient checkpointing + SFT

param_count = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  Parameters: {param_count:.2f}B")
print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 9. TRAINING ARGS ─────────────────────────────────────────────────────────

sft_cfg = SFTConfig(
    output_dir                  = cfg.output_dir,
    num_train_epochs            = cfg.num_epochs,
    per_device_train_batch_size = cfg.per_device_bs,
    per_device_eval_batch_size  = cfg.per_device_bs,
    gradient_accumulation_steps = cfg.grad_accum,
    gradient_checkpointing      = True,
    learning_rate               = cfg.lr,
    lr_scheduler_type           = cfg.lr_scheduler,
    warmup_ratio                = cfg.warmup_ratio,
    weight_decay                = cfg.weight_decay,
    max_grad_norm               = cfg.max_grad_norm,
    bf16                        = cfg.bf16,
    fp16                        = cfg.fp16,
    logging_steps               = cfg.logging_steps,
    evaluation_strategy         = "steps",
    eval_steps                  = cfg.eval_steps,
    save_strategy               = "steps",
    save_steps                  = cfg.save_steps,
    save_total_limit            = cfg.save_total_limit,
    load_best_model_at_end      = cfg.load_best_model,
    metric_for_best_model       = cfg.metric_for_best,
    report_to                   = "wandb" if cfg.use_wandb else "none",
    seed                        = cfg.seed,
    dataset_text_field          = "text",
    max_seq_length              = cfg.max_seq_len,
    packing                     = False,   # keeps samples separate; set True to speed up
)

# ── 10. CUSTOM CALLBACK: live loss logging ────────────────────────────────────

class LiveLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            step  = state.global_step
            total = state.max_steps
            parts = [f"step {step}/{total}"]
            for k in ("loss","eval_loss","learning_rate"):
                if k in logs:
                    parts.append(f"{k}={logs[k]:.5f}" if "loss" in k else f"{k}={logs[k]:.2e}")
            print("  [LOG]", " | ".join(parts))

# ── 11. TRAINER ──────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model           = model,
    args            = sft_cfg,
    train_dataset   = ds["train"],
    eval_dataset    = ds["val"],
    tokenizer       = tokenizer,
    callbacks       = [
        LiveLogger(),
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
)

# ── 12. TRAIN ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Starting full fine-tuning …")
print("="*60)
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} min")

# ── 13. SAVE ─────────────────────────────────────────────────────────────────

trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print(f"Model saved → {cfg.output_dir}")

if cfg.use_wandb:
    wandb.finish()

# ── 14. QUICK INFERENCE CHECK ────────────────────────────────────────────────

print("\n--- Inference sanity check ---")

model.eval()
sample_row = raw[0]
prompt = build_chat_prompt(sample_row, tokenizer, include_response=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens   = 256,
        temperature      = 0.7,
        top_p            = 0.9,
        do_sample        = True,
        repetition_penalty = 1.1,
        eos_token_id     = tokenizer.eos_token_id,
        pad_token_id     = tokenizer.pad_token_id,
    )

generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Prompt excerpt:", prompt[:200], "…")
print("\nGenerated:\n", generated)
print("\nReference:\n", sample_row.get("response",""))
