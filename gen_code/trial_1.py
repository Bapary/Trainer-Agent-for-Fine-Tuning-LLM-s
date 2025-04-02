
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import datasets
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
REF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
OUTPUT_DIR = "./trained_model"
PRECISION = "bf16"  # Change to "fp16" if desired
MAX_TOKENS = 4096

# Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Set chat_template if required (replace "your_chat_template_here" with the actual template)
tokenizer.chat_template = "your_chat_template_here"

# Enable Gradient Checkpointing for memory optimization
model.gradient_checkpointing_enable()

# Load Dataset
train_dataset = datasets.load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="trained_model", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()


trainer.save_model(OUTPUT_DIR)

logging.info("Training completed and model saved.")
