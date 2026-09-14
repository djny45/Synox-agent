"""Deterministic smoke evaluation for a trained SYNXO model."""
from pathlib import Path
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "artifacts/synxo-nano"
TESTS = [
    "What is your purpose?",
    "Explain token efficiency in one sentence.",
    "What is an AI model?",
]

if not MODEL_DIR.exists():
    print("FAIL: no trained SYNXO model artifact exists.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()

results = []
for prompt in TESTS:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    results.append({"prompt": prompt, "answer": answer, "non_empty": bool(answer)})

passed = all(item["non_empty"] for item in results)
report = {"model": "SYNXO Nano", "tests": results, "passed": passed}
(ROOT / "evaluation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if passed else 1)
