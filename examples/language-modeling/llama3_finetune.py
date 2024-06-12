import torch
from transformers import AutoTokenizer
from optimum.tpu import AutoModelForCausalLM
from datasets import load_dataset
from optimum.tpu import get_fsdp_config, use_fsdp_v2, get_fsdp_training_args
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import TrainerCallback
from pathlib import Path

MAX_LENGTH = 2048
BATCH_SIZE = 1
OUTPUT_DIR = f"/home/yirmibesogluz/llama3_outdir/streaming_{BATCH_SIZE}_{MAX_LENGTH}/"

use_fsdp_v2()

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add custom token for padding Llama
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

print("***Model loading...")
model = AutoModelForCausalLM.from_pretrained(model_id, sequence_length=MAX_LENGTH)

print("***Model loaded")

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

print("***Dataset loading...")
data = load_dataset("vngrs-ai/vngrs-web-corpus", streaming=True)
shuffled_dataset = data.shuffle(seed=25)
train_data = shuffled_dataset.map(encode, batched=True)
print("***Dataset loaded")

print("***fsdp_training_args loaded")
fsdp_training_args = get_fsdp_training_args(model)

class ModelSaveGCSCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print("***Entered callback")
        output_path = Path(OUTPUT_DIR)
        print(f"***Directory: {list(output_path.iterdir())}")


print("***Trainer loaded")
trainer = Trainer(
    model=model,
    train_dataset=train_data["train"].with_format("torch"),
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        save_steps=1,
        save_total_limit=1,
        max_steps=1000000,
        output_dir=OUTPUT_DIR,
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last=True,  # Required by FSDP v2 and SPMD.
        **fsdp_training_args,
        callbacks=[ModelSaveGCSCallback()],
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("***Training loaded")
trainer.train()
