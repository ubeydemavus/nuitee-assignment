import logging
import json
import random
import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the lowest level to capture
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Ensures logs go to the console
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


train_path = "./datasets/finetune_dataset.json"

with open(train_path, "r") as f:
    data = json.load(f)

logging.info('Dataset size:' + str(len(data)))

random.shuffle(data)
logging.info('Dataset is shuffled...')

train_dataset = data[:int(len(data)*0.9)]
test_dataset = data[int(len(data)*0.9):]

logging.info('Dataset is splitted...')

model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5") 

# use it for better performance, it mimics original implementation but it's less memory efficient
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

model.to(device)

# Training Configs
num_steps = 500 # this settings will train for one epoch.
batch_size = 8
data_size = len(train_dataset)
num_batches = data_size // batch_size
num_epochs = max(1, num_steps // num_batches)

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear", #cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    save_steps = 100,
    save_total_limit=10,
    dataloader_num_workers = 0,
    use_cpu = False,
    report_to="none",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

trainer.train()

### ONCE TRANING IS DONE,
### The trained model can be used via "model = GLiNER.from_pretrained("models/checkpoint-100", load_tokenizer=True)", change "models/checkpoint-100" as accordingly.