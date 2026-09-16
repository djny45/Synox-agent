"""Train the SYNXO nano model with a small LoRA adapter."""
from pathlib import Path
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

train = load_dataset("json", data_files=str(ROOT / "data/train.jsonl"), split="train")
valid = load_dataset("json", data_files=str(ROOT / "data/validation.jsonl"), split="train")

tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

args = TrainingArguments(
    output_dir=str(ROOT.parent / cfg["output_dir"]),
    num_train_epochs=cfg["num_train_epochs"],
    learning_rate=cfg["learning_rate"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    warmup_ratio=cfg["warmup_ratio"],
    weight_decay=cfg["weight_decay"],
    logging_steps=cfg["logging_steps"],
    eval_strategy=cfg["eval_strategy"],
    save_strategy=cfg["save_strategy"],
    load_best_model_at_end=cfg["load_best_model_at_end"],
    report_to="none",
    seed=cfg["seed"],
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=valid,
    peft_config=peft,
    tokenizer=tokenizer,
    max_seq_length=cfg["max_seq_length"],
)

trainer.train()
trainer.save_model(str(ROOT.parent / cfg["output_dir"]))
tokenizer.save_pretrained(str(ROOT.parent / cfg["output_dir"]))
print("SYNXO training finished. Evaluation is required before deployment.")
