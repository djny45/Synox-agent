"""Evaluate a trained SYNXO LoRA adapter against deterministic behavior checks."""
from pathlib import Path
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import yaml

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "artifacts/synxo-nano"
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

TESTS = [
    ("What is your purpose?", ["compact", "efficient"]),
    ("Explain token efficiency in one sentence.", ["token"]),
    ("What is an AI model?", ["trained", "model"]),
    ("What should you do if you are uncertain?", ["uncertainty", "invent"]),
    ("Define LoRA briefly.", ["lora", "parameter"]),
]

if not MODEL_DIR.exists():
    print("FAIL: no trained SYNXO model artifact exists.")
    sys.exit(1)

# train.py saves a PEFT adapter, so evaluation must load the base model first.
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
base_model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

results = []
for prompt, required_terms in TESTS:
    messages = [
        {"role": "system", "content": "You are SYNXO, a compact AI model focused on precise, efficient assistance."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    normalized = answer.lower()
    matched = [term for term in required_terms if term.lower() in normalized]
    passed = bool(answer) and bool(matched)
    results.append({
        "prompt": prompt,
        "answer": answer,
        "non_empty": bool(answer),
        "required_terms": required_terms,
        "matched_terms": matched,
        "passed": passed,
    })

passed = all(item["passed"] for item in results)
report = {
    "model": "SYNXO Nano",
    "base_model": cfg["base_model"],
    "evaluation_type": "deterministic_behavior_gate",
    "tests": results,
    "passed": passed,
}
(ROOT / "evaluation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if passed else 1)
