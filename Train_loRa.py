from __future__ import annotations

import logging
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = (
    "You are a drug-information assistant. Answer questions about medications and "
    "clinical trials, their uses, mechanisms, interactions, and benefits from the "
    "reference material provided. Cite the sources you draw from. If the material "
    "does not contain the answer, say so plainly and do not speculate. Do not give "
    "individualized medical advice or dosing. Where it helps, provide a chart that "
    "supports the information for the use case. Include a disclaimer that you are "
    "not diagnosing or treating, and that those decisions belong to a licensed "
    "clinician or pharmacist."
)

assert torch.cuda.is_available(), "torch cannot see the GPU"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger("train_lora")
LOG.info("device: %s", torch.cuda.get_device_name(0))

MODEL_ID = "google/gemma-4-12B-it"   # instruct checkpoint; "google/gemma-4-12B" for base
INSTRUCTIONS = "pairs.jsonl"
OUTPUT_DIR = "gemma-4-12b-drugbot-lora"
SEQ_LEN = 512
LEARNING_RATE = 2e-4
TRAIN_ROWS = 2000  

torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def create_conversation(sample: dict) -> dict:
    # adjust the two field names to match your JSONL schema (prompt/response assumed)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"].strip()},
            {"role": "assistant", "content": sample["response"].strip()},
        ]
    }
def build_dataset():
    split = "train" if TRAIN_ROWS is None else f"train[:{TRAIN_ROWS}]"
    raw = load_dataset("json", data_files=INSTRUCTIONS, split=split)
    ds = raw.map(create_conversation, remove_columns=raw.column_names, batched=False)
    return ds.train_test_split(test_size=0.05,seed=42)
def build_model():
    model = AutoModelForCausalLM .from_pretrained(
    MODEL_ID,
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    )
    return model, tokenizer
    
def main() -> None:
    dataset = build_dataset()
    c_train =  len(dataset["train"])
    c_test =  len(dataset['test'])
    LOG.info(f"train rows: {c_train} eval rows: {c_test}") 
    model, tokenizer = build_model()
    processor= AutoProcessor.from_pretrained(MODEL_ID)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        
    )  
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=SEQ_LEN,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,                              # <-- add
        gradient_checkpointing_kwargs={"use_reentrant": False},  # <-- add
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_grad_norm=0.3,                 # from the QLoRA paper
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        push_to_hub=False,                 # keep the medical adapter local
        use_liger_kernel=False,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer ,
    )
    trainer.train()
    trainer.save_model()
    LOG.info("saved adapter to %s", OUTPUT_DIR)
    hist = pd.DataFrame(trainer.state.log_history) # TRL appends every step + every epoch eval here. This replaces your old `losses` list.
    hist.to_csv("gemm4lora_log_history.csv", index=False)
    train_log = hist[hist["loss"].notna()].copy() # rows carrying a training loss vs rows carrying an eval loss
    eval_log = hist[hist["eval_loss"].notna()].copy()
    
    train_log["epoch_i"] = train_log["epoch"].round().astype(int)
    eval_log["epoch_i"] = eval_log["epoch"].round().astype(int)
    
    table = pd.DataFrame({
    "Epoch": eval_log["epoch_i"].values,
    "Training Loss": train_log.groupby("epoch_i")["loss"].last().reindex(eval_log["epoch_i"]).values,
    "Validation Loss": eval_log["eval_loss"].values,
    })
    for key, label in [("eval_entropy", "Entropy"),
                    ("eval_mean_token_accuracy", "Mean Token Accuracy"),
                    ("num_tokens", "Num Tokens")]:
        if key in eval_log:
            table[label] = eval_log[key].values

    table.to_csv("gemma4_lora_metrics.csv", index=False)
    LOG.info("per-epoch metrics:\n%s", table.to_string(index=False))

    # loss curves 
    plt.figure()
    plt.plot(train_log["step"], train_log["loss"], label="training loss")
    plt.plot(eval_log["step"], eval_log["eval_loss"], marker="o", label="validation loss")
    plt.xlabel("step")
    plt.ylabel("loss (cross entropy)")
    plt.title("Gemma 4 LoRA training loss")
    plt.legend()
    plt.savefig("Gemma4_Lora_training_loss.png")
    LOG.info("saved Gemma4_Lora_training_loss.png")

    # validation token accuracy
    if "eval_mean_token_accuracy" in eval_log:
        plt.figure()
        plt.plot(eval_log["epoch_i"], eval_log["eval_mean_token_accuracy"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("mean token accuracy")
        plt.title("Gemma 4 LoRA validation token accuracy")
        plt.savefig("Gemma4_Lora_token_accuracy.png")
        LOG.info("saved Gemma4_Lora_token_accuracy.png")


if __name__ == "__main__":
    main()



