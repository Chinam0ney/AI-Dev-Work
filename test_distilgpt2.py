from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model     = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")

inputs  = tokenizer("Hello, brother!", return_tensors="pt").to("cuda")
out_ids = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
