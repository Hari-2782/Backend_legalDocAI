from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL_NAME = "t5-base"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Load your dataset built by dataset_builder.py
dataset = load_dataset("json", data_files="dataset.json")

def preprocess(example):
    model_inputs = tokenizer(example["question"], max_length=512, truncation=True)
    labels = tokenizer(example["answer"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training args
args = Seq2SeqTrainingArguments(
    output_dir="./model_lora",
    evaluation_strategy="steps",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation", tokenized_dataset["train"]),
    tokenizer=tokenizer
)

trainer.train()
