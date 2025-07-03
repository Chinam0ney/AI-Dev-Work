from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model     = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 2. Apply LoRA adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# 3. Load & preprocess data
dataset = load_dataset("json", data_files="chat_data.json", split="train")
def preprocess(ex):
    inp  = tokenizer(ex["prompt"],  return_tensors="pt").input_ids[0]
    lbls = tokenizer(ex["response"], return_tensors="pt").input_ids[0]
    return {"input_ids": inp, "labels": lbls}
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=1,
    fp16=True,
)

# 5. Trainer & train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# 6. Save your new Nova 2.0
model.save_pretrained("omega_lora")
tokenizer.save_pretrained("omega_lora")
