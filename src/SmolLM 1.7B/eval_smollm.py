"""
Post-training evaluation for SmolLM2-1.7B fine-tuned on Recast-30k
===================================================================
Run AFTER smollm_finetune.py has completed.
Computes: constraint pass rate, ROUGE-L, BERTScore, per-sample JSON output.

Usage:
  python eval_smollm.py \
      --model_dir ./smollm2_finetuned \
      --test_file ./smollm2_finetuned/test_split.json \
      --n_samples 200 \
      --out_file   eval_results.json
"""

import json, re, argparse, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from evaluate import load as load_metric

# ── CONSTRAINT CHECKER ───────────────────────────────────────────────────────

def check_constraints(response: str, constraints) -> dict[str, str]:
    if not constraints:
        return {}
    if isinstance(constraints, str):
        constraints = [constraints]

    results = {}
    for c in constraints:
        cl = c.lower()
        rl = response.lower()

        # word count
        if "word" in cl and re.search(r"\d+", cl):
            nums = [int(x) for x in re.findall(r"\d+", cl)]
            wc   = len(response.split())
            if any(k in cl for k in ("fewer","less","under","at most")):
                results[c] = "pass" if wc < max(nums) else "fail"
            elif any(k in cl for k in ("more","at least","over","minimum")):
                results[c] = "pass" if wc >= min(nums) else "fail"
            elif "exactly" in cl:
                results[c] = "pass" if wc == nums[0] else "fail"
            else:
                results[c] = "unknown"

        # sentence count
        elif "sentence" in cl and re.search(r"\d+", cl):
            nums = [int(x) for x in re.findall(r"\d+", cl)]
            sc   = len(re.split(r'[.!?]+', response.strip()))
            results[c] = "pass" if sc <= max(nums) else "fail"

        # tone
        elif any(k in cl for k in ("formal","professional","polite")):
            informal = ["gonna","wanna","ain't","stuff","kinda","dunno","yeah","nope"]
            results[c] = "pass" if not any(w in rl for w in informal) else "fail"

        # structure
        elif any(k in cl for k in ("bullet","list","numbered")):
            results[c] = "pass" if any(m in response for m in ["\n-","\n•","\n*","\n1."]) else "fail"

        # json
        elif "json" in cl:
            try:
                json.loads(response.strip())
                results[c] = "pass"
            except Exception:
                results[c] = "fail"

        # include keyword
        elif re.search(r"(?:include|mention|contain)\s+[\"']?(\w[\w\s]*)[\"']?", cl):
            m = re.search(r"(?:include|mention|contain)\s+[\"']?(\w[\w\s]+?)[\"']?(?:\.|$)", cl)
            if m:
                kw = m.group(1).strip()
                results[c] = "pass" if kw in rl else "fail"
            else:
                results[c] = "unknown"

        # avoid keyword
        elif re.search(r"(?:avoid|do not use|never use|without)\s+[\"']?(\w[\w\s]*)[\"']?", cl):
            m = re.search(r"(?:avoid|do not use|never use|without)\s+[\"']?(\w[\w\s]+?)[\"']?(?:\.|$)", cl)
            if m:
                kw = m.group(1).strip()
                results[c] = "pass" if kw not in rl else "fail"
            else:
                results[c] = "unknown"

        else:
            results[c] = "unknown"

    return results

# ── PROMPT BUILDER (same as training) ────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise assistant. Read the instruction and satisfy EVERY constraint listed. "
    "Do not skip or partially satisfy any constraint."
)

def build_prompt(row: dict, tokenizer) -> str:
    instruction = row.get("instruction", row.get("prompt", ""))
    constraints = row.get("constraints", [])
    context     = row.get("context", "") or ""
    user_content = instruction
    if context:
        user_content = f"Context:\n{context}\n\n{user_content}"
    if constraints:
        if isinstance(constraints, list):
            user_content += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        else:
            user_content += f"\n\nConstraints:\n- {constraints}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(args):
    print(f"Loading model from {args.model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # for batch inference

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading test data from {args.test_file} …")
    with open(args.test_file) as f:
        test_rows = json.load(f)

    # test_split.json stores {"text": full_prompt_with_response}
    # We need original rows for inference; load from the raw dataset too
    raw_path = Path(args.raw_data) if args.raw_data else None

    # If you saved raw rows alongside test_split, load them; otherwise
    # we reconstruct what we can from the stored text.
    rows = test_rows[:args.n_samples]
    print(f"Evaluating {len(rows)} samples …")

    rouge_metric  = load_metric("rouge")
    bertscore_met = load_metric("bertscore")

    results = []
    all_preds, all_refs = [], []

    for i, row in enumerate(rows):
        # row may be {"text": "..."} (from saved split) or a raw dict
        if "instruction" not in row and "text" in row:
            # We can still do constraint checks but skip generation;
            # to generate you need the raw row — load raw data via --raw_data
            print(f"  [warn] row {i} has no 'instruction' key; skipping generation.")
            continue

        prompt    = build_prompt(row, tokenizer)
        reference = row.get("response", row.get("output", ""))
        constraints = row.get("constraints", [])

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens     = 256,
                temperature        = 0.7,
                top_p              = 0.9,
                do_sample          = True,
                repetition_penalty = 1.1,
                eos_token_id       = tokenizer.eos_token_id,
                pad_token_id       = tokenizer.pad_token_id,
            )
        latency = time.time() - t0
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        checks  = check_constraints(generated, constraints)
        passed  = sum(1 for v in checks.values() if v == "pass")
        total   = len(checks)
        checkable = sum(1 for v in checks.values() if v != "unknown")
        rate    = passed / checkable if checkable else None

        all_preds.append(generated)
        all_refs.append(reference)

        results.append({
            "sample_id"         : i,
            "prompt"            : prompt[:300],
            "reference"         : reference,
            "generated"         : generated,
            "latency_s"         : round(latency, 3),
            "output_tokens"     : out.shape[1] - inputs["input_ids"].shape[1],
            "constraint_checks" : checks,
            "constraints_passed": passed,
            "constraints_total" : total,
            "pass_rate"         : round(rate, 3) if rate is not None else None,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)} done")

    # ── CORPUS-LEVEL METRICS ──────────────────────────────────────────────────
    print("\nComputing ROUGE …")
    rouge_out = rouge_metric.compute(predictions=all_preds, references=all_refs, use_stemmer=True)

    print("Computing BERTScore (this may take a minute) …")
    bs_out = bertscore_met.compute(predictions=all_preds, references=all_refs, lang="en")
    mean_f1 = sum(bs_out["f1"]) / len(bs_out["f1"])

    pass_rates = [r["pass_rate"] for r in results if r["pass_rate"] is not None]
    avg_pass   = sum(pass_rates) / len(pass_rates) if pass_rates else None

    summary = {
        "n_evaluated"           : len(results),
        "avg_constraint_pass_rate": round(avg_pass, 4) if avg_pass else None,
        "rouge1"                : round(rouge_out["rouge1"], 4),
        "rouge2"                : round(rouge_out["rouge2"], 4),
        "rougeL"                : round(rouge_out["rougeL"], 4),
        "bertscore_f1"          : round(mean_f1, 4),
        "avg_latency_s"         : round(sum(r["latency_s"] for r in results)/len(results), 3),
    }

    print("\n── Evaluation Summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")

    output = {"summary": summary, "results": results}
    with open(args.out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results → {args.out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  default="./smollm2_finetuned")
    p.add_argument("--test_file",  default="./smollm2_finetuned/test_split.json")
    p.add_argument("--raw_data",   default="RECAST-30K.json",
                   help="Original dataset file (needed to re-build prompts from saved test split)")
    p.add_argument("--n_samples",  type=int, default=200)
    p.add_argument("--out_file",   default="eval_results_smollm2_finetuned.json")
    main(p.parse_args())
