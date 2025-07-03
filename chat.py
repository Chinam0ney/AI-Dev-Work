from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("omega_lora")
model     = AutoModelForCausalLM.from_pretrained("omega_lora").to("cuda")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("exit","quit"):
        break
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    replies = model.generate(**inputs, max_new_tokens=50)
    print("Omega:", tokenizer.decode(replies[0], skip_special_tokens=True))
