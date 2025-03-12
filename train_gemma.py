import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ✅ Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {device}")

# ✅ Load Gemma model with 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_name = "google/gemma-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# ✅ Prepare model for LoRA fine-tuning (fixes "does not require grad" error)
model = prepare_model_for_kbit_training(model)

# ✅ Apply PEFT LoRA adapters
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ✅ Unfreeze **only** LoRA adapter weights (fixes gradient issue)
for name, param in model.named_parameters():
    if "lora" in name:  # ✅ Only apply to LoRA adapter weights
        param.requires_grad = True
    else:
        param.requires_grad = False  # Keep all other weights frozen

# ✅ Verify trainable parameters
model.print_trainable_parameters()

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load Dataset
dataset_file = "train_data_fixed.jsonl"
raw_dataset = load_dataset("json", data_files=dataset_file)

# ✅ Tokenization function
def preprocess_function(examples):
    inputs = [str(ex) if isinstance(ex, str) else "" for ex in examples["prompt"]]
    targets = [str(ex) if isinstance(ex, str) else "" for ex in examples["completion"]]

    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

    model_inputs["input_ids"] = model_inputs["input_ids"].tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"].tolist()
    model_inputs["labels"] = labels["input_ids"].tolist()
    return model_inputs

# ✅ Apply tokenization
tokenized_datasets = raw_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "completion"])

# ✅ Training Arguments (Optimized for Low VRAM)
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    fp16=True,
    weight_decay=0.0,  # ✅ Fixes weight decay on quantized models
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False
)

# ✅ Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer
)

# 🚀 Start Training
print("🚀 Starting training with PEFT LoRA...")
trainer.train()
# Save the fine-tuned model after training
model.save_pretrained("D:/Gemma/fine_tuned_gemma")
tokenizer.save_pretrained("D:/Gemma/fine_tuned_gemma")
