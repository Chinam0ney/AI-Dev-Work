import torch
# Force eager execution: disable torch.compile and Torch Dynamo/Inductor
try:
    torch.compile = lambda model, **kwargs: model
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.disable()
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 2. Load & preprocess data with masked loss and fixed-length padding
raw_dataset = load_dataset("json", data_files="chat_data.json", split="train")
print(f"Loaded {len(raw_dataset)} examples from chat_data.json")

MAX_LEN = 64

def preprocess(example):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False).input_ids
    response_ids = tokenizer(example["response"], add_special_tokens=False).input_ids
    input_ids = prompt_ids + [tokenizer.eos_token_id] + response_ids
    labels = [-100] * (len(prompt_ids) + 1) + response_ids
    attention_mask = [1] * len(input_ids)
    # Truncate and pad to MAX_LEN
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
    else:
        pad_len = MAX_LEN - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

print("Preprocessing examples with masked labels and fixed-length padding...")
dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
print(f"After preprocessing, sample: {dataset[0]}")

# 3. Apply LoRA adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=1e-5,
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    report_to=[],
    save_total_limit=1,
    fp16=False,
    disable_tqdm=False
)

# 5. Trainer & train with default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator
)

print("Starting LoRA fine-tuning with masked loss and fixed-length inputs...")
trainer.train()
print("Training complete. Saving model...")

# 6. Save fine-tuned model and tokenizer
model.save_pretrained("omega_lora")
tokenizer.save_pretrained("omega_lora")
print("Model and tokenizer saved to omega_lora/")
