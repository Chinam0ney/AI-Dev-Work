# Omega LoRA Fine-Tuning

This repository contains the code and data for fine-tuning DistilGPT-2 with LoRA adapters to create **Omega**, a lightweight conversational AI.

## Setup

1. **Clone repo**  
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create Conda environment**  
   ```bash
   conda create -n omega python=3.8 -y
   conda activate omega
   ```

3. **Install dependencies**  
   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -y
   pip install transformers datasets accelerate peft
   ```

4. **Prepare data**  
   - Edit `chat_data.json` with your prompt/response pairs.

## Fine-Tuning

Run the LoRA training script:
```bash
accelerate launch train_lora.py
```
Model and tokenizer will be saved to `omega_lora/`.

## Chat

Interact with the model:
```bash
python chat.py
```

## Next Steps

- Expand `chat_data.json` to more examples.
- Tune hyperparameters in `train_lora.py`.
- Scale up on cloud GPUs for larger models and datasets.
