from transformers import AutoTokenizer, AutoModelForCausalLM

def chat():
    tokenizer = AutoTokenizer.from_pretrained("omega_lora")
    model = AutoModelForCausalLM.from_pretrained("omega_lora")
    device = "cuda" if model.device.type == "cuda" else "cpu"
    model.to(device)

    system_prompt = "You are Omega, a helpful, friendly conversational AI."
    conversation = system_prompt + tokenizer.eos_token

    print("Chat with Omega (type 'exit' or 'quit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Omega: Goodbye!")
            break

        conversation += "<|user|>" + user_input + tokenizer.eos_token
        inputs = tokenizer(conversation, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print("Omega:", reply)
        conversation += reply + tokenizer.eos_token

if __name__ == "__main__":
    chat()
