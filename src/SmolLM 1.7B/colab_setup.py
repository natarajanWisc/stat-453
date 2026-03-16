# ═══════════════════════════════════════════════════════════════════════════
#  CELL 0 — Run once at the top of your Colab session
# ═══════════════════════════════════════════════════════════════════════════

# 1. Check GPU
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                         "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", result.stdout.strip())

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# 2. Install dependencies
# Pinned versions tested against SmolLM2 + TRL 0.10
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.44.2",
    "datasets>=2.20",
    "accelerate>=0.33",
    "trl==0.10.1",
    "peft>=0.12",
    "evaluate",
    "rouge_score",
    "bert_score",
    "bitsandbytes",    # needed for bnb quantization helpers even in full FT
    "wandb",
    "sentencepiece",
    "protobuf",
], check=True)

print("\nAll dependencies installed ✓")

# 3. Log in to HuggingFace (needed to download SmolLM2-Instruct)
from huggingface_hub import notebook_login
notebook_login()   # paste your HF token with read access

# 4. (Optional) Log in to W&B
import wandb
wandb.login()      # paste your W&B API key

# ═══════════════════════════════════════════════════════════════════════════
#  CELL 1 — Upload the dataset
# ═══════════════════════════════════════════════════════════════════════════
from google.colab import files
uploaded = files.upload()   # select RECAST-30K.json from your Mac
# File will appear at /content/RECAST-30K.json

# ═══════════════════════════════════════════════════════════════════════════
#  CELL 2 — Upload and run the training script
# ═══════════════════════════════════════════════════════════════════════════
# Upload smollm_finetune.py the same way, then:
# !python smollm_finetune.py

# ─── OR paste the full script inline into a notebook cell ───────────────
# (copy-paste the contents of smollm_finetune.py directly)

# ═══════════════════════════════════════════════════════════════════════════
#  CELL 3 — (After training) Save checkpoint to Google Drive
# ═══════════════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import shutil, os
src = "/content/smollm2_finetuned"
dst = "/content/drive/MyDrive/smollm2_finetuned"
shutil.copytree(src, dst, dirs_exist_ok=True)
print("Saved to Google Drive ✓")

# ═══════════════════════════════════════════════════════════════════════════
#  CELL 4 — (After training) Run evaluation
# ═══════════════════════════════════════════════════════════════════════════
# Upload eval_smollm.py, then:
# !python eval_smollm.py \
#     --model_dir  /content/smollm2_finetuned \
#     --test_file  /content/smollm2_finetuned/test_split.json \
#     --raw_data   /content/RECAST-30K.json \
#     --n_samples  200 \
#     --out_file   eval_results_finetuned.json

# ═══════════════════════════════════════════════════════════════════════════
#  EXPECTED MEMORY PROFILE (rough estimates)
# ═══════════════════════════════════════════════════════════════════════════
#
#  SmolLM2-1.7B in bf16:
#    Model weights               ~3.4 GB
#    Optimizer states (AdamW)    ~6.8 GB  (2x weights for full FT)
#    Gradients                   ~3.4 GB
#    Activations (bs=4, gc on)   ~4-6 GB
#    ─────────────────────────────────────
#    Total (approx)              ~18-20 GB   ← fits comfortably in A100 (40GB)
#                                            ← fits in L4 (24GB) with bs=2
#
#  If you see OOM on L4:
#    - Set per_device_bs = 2, grad_accum = 8  (same effective batch = 16)
#    - Or set max_seq_len = 512
#
# ═══════════════════════════════════════════════════════════════════════════
#  RUNTIME ESTIMATE
# ═══════════════════════════════════════════════════════════════════════════
#
#  25,000 train samples × 3 epochs on A100:
#    ~2.5 - 3.5 hours  (packing=False)
#    ~1.5 - 2.0 hours  (packing=True, if samples are short)
#
#  On L4 (24GB):
#    ~4 - 5 hours
