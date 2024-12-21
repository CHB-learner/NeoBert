from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import Dataset, DataLoader
import json
from datasets import load_metric
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="516_train"
)
wandb.init(project='516_train', entity='1945626852')


class myDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)


def make_data(args, tokenizer):
    # train_json = [json.loads(line) for line in open(args.train_data_path).readlines()]
    # val_json = [json.loads(line) for line in open(args.val_data_path).readlines()]
    # test_json = [json.loads(line) for line in open(args.val_data_path).readlines()]
    train_json = json.load(open(args.train_data_path))
    test_json = json.load(open(args.val_data_path))
    val_json = json.load(open(args.val_data_path))
    label2id = json.load(open(args.label_path))

    def process_data(data):
        features = []
        for example in data:
            feature = tokenizer(
                example["text"],
                padding="max_length",
                max_length=args.max_seq_length,
                return_token_type_ids=True,
                truncation=True
            )
            feature["labels"] = label2id[example["label"]]
            features.append(feature)

        return features

    train_features = process_data(train_json)
    val_features = process_data(val_json)
    test_features = process_data(test_json)
    return myDataset(train_features), myDataset(val_features), myDataset(test_features)


model_path = "./bert-base-uncased/"   #https://huggingface.co/google-bert/bert-base-uncased
config = AutoConfig.from_pretrained(model_path)
config.num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

metric = load_metric("./accuracy.py")


from transformers import EvalPrediction
import numpy as np
def compute_metrics(p: EvalPrediction):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    # labels = np.argmax(labels, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return results

'''
report_to：指定报告日志的目的地，这里设置为 "none" 表示不报告日志。
do_train：是否执行训练。
do_eval：是否执行评估。
do_predict：是否执行预测。
output_dir：输出模型和结果的目录。
per_device_train_batch_size：每个设备的训练批次大小。
per_device_eval_batch_size：每个设备的评估批次大小。
num_train_epochs：训练的总轮数。
learning_rate：初始学习率。
evaluation_strategy：评估策略，这里设置为 "steps" 表示按步数评估。
eval_steps：每隔多少步进行一次评估。
save_strategy：保存模型的策略，这里设置为 "steps" 表示按步数保存。
save_steps：每隔多少步保存一次模型。
save_total_limit：保存模型的最大数量。
load_best_model_at_end：训练结束时是否加载最佳模型。
metric_for_best_model：用于选择最佳模型的评估指标。
remove_unused_columns：是否移除未使用的列。
overwrite_output_dir：是否覆盖输出目录。
eval_accumulation_steps：评估累积步数。
fp16：是否使用混合精度训练。
logging_steps：每隔多少步记录一次日志。
dataloader_num_workers：数据加载器的工作进程数。
'''

training_args = TrainingArguments(
        report_to="wandb",  # 2485438bb42961f869bc4908951674962e75617b 2485438bb42961f869bc4908951674962e75617b
        do_train=True,
        do_eval=True,
        do_predict=True,
        output_dir="./output/",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=1500,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=30,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=True,
        overwrite_output_dir=True,
        eval_accumulation_steps=5,
        fp16=False,   # True
        logging_steps=10,
        dataloader_num_workers = 8
    )

class Args:
    def __init__(self):
        self.train_data_path = "./data/train.json"
        self.val_data_path = "./data/test.json"
        self.label_path = "./data/label2id.json"
        self.max_seq_length = 128

args = Args()

train_dataset, eval_dataset, test_dataset = make_data(args, tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

trainer.train()