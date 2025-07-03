# Nova 2.0 – Mini Chatbot with LoRA

This repository showcases a lean version of “Nova”, a GPT‐style conversational AI fine‐tuned with LoRA adapters.

## Project Structure

- `environment.yml` &ndash; Conda environment spec
- `check_gpu.py` &ndash; Verify GPU availability
- `test_distilgpt2.py` &ndash; Load and generate sample text
- `chat_data.json` &ndash; Tiny chat dataset
- `train_lora.py` &ndash; Script to fine‐tune `distilgpt2` with LoRA
- `chat.py` &ndash; Interactive chat client using the fine‐tuned model

## Getting Started

1. Clone this repo and navigate into it.
2. Create and activate the Conda env:
   ```bash
   conda env create -f environment.yml
   conda activate tinychat
   ```
3. Verify GPU:
   ```bash
   python check_gpu.py
   ```
4. Test inference:
   ```bash
   python test_distilgpt2.py
   ```
5. Train your mini‐chatbot:
   ```bash
   accelerate launch train_lora.py
   ```
6. Chat with your model:
   ```bash
   python chat.py
   ```

Feel free to fork and extend this project for your own experiments!
