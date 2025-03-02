from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, TrainingArguments, Trainer, EvalPrediction
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, recall_score, confusion_matrix 
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from datasets import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss, BCELoss, Softmax
from Bio import SeqIO
from torch import nn
import torch
import pandas as pd
import datasets
import transformers
import numpy as np
import threading
import argparse
import torch
import csv
import re
import os
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {device}")
torch.cuda.empty_cache()
print("Cache emptied")

tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

num_labels = 3
model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        num_labels = 3,
    )
model = model.to(device)

cpu_threads =2

def csv_to_dict(csv_file):
    # Open the CSV file
    with open(csv_file, mode='r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Extract column headers
        headers = next(reader)
        # Initialize an empty list to store dictionaries
        data = []
        # Iterate through each row and create a dictionary
        for row in reader:
            # Create a dictionary for the current row
            row_dict = {header: value for header, value in zip(headers, row)}
            # Append the dictionary to the data list
            data.append(row_dict)
    return data

# Example usage
csv_file = "Generated Sample data.csv"
data = csv_to_dict(csv_file)
for item in data:
    # Convert the string representation of labels into a list
    labels = eval(item.pop('labels'))
    # Add separate keys for each label with their corresponding boolean values
    for label in ['SNV', 'INDEL', 'DUP']:
        item[label] = label in labels

# Load dataset
dataset = pd.DataFrame(data)

datasets = Dataset.from_pandas(dataset).train_test_split(test_size=0.2, seed=42)

train_dataset = datasets["train"]
val_dataset = datasets["test"]

labels = [label for label in train_dataset.features.keys() if label not in ['Sequence']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


def preprocess_data(examples):
    snv_count=0
    # take a batch of texts
    text = examples["Sequence"]

    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)

    # Add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}

    # Create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))

    # Fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    
    return encoding

batch_size = 4
metric_name = "f1"
args = TrainingArguments(
    "trained_dnabert_final",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_eval_batch_size=batch_size,
    num_train_epochs=7,
    weight_decay=0.01,
    do_train=True,
    do_eval=True,
    logging_steps= 100,
    metric_for_best_model=metric_name,
    logging_strategy='epoch'
    #push_to_hub=True,
)

encoded_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
encoded_dataset = encoded_dataset.shuffle(seed=42)
encoded_val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)
encoded_val_dataset = encoded_dataset.shuffle(seed=42)

encoded_dataset.set_format("torch")
encoded_val_dataset.set_format("torch")

input_ids = encoded_dataset['input_ids'][0].unsqueeze(0).to(device)
labels = encoded_dataset[0]['labels'].unsqueeze(0).to(device)
outputs = model(input_ids=input_ids, labels=labels)
print(outputs)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    reca = recall_score(y_true, y_pred, average = 'weighted')
    class_report = classification_report(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'weighted')
    
    f1_micro_average = f'{f1_micro_average:.4f}'
    reca = f'{reca:.4f}'
    accuracy = f'{accuracy:.4f}'
    precision= f'{precision:.4f}'
    metrics = {'f1': f1_micro_average,
               'recall' : reca,
               'accuracy': accuracy,
               'precision':precision}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Move labels tensor to the same device as logits (CUDA)

        # Forward pass
        outputs = model(**inputs)
        logits = (outputs.logits).to(device)

        
        num_classes = logits.size(-1)
        batch_size = logits.size(0)

        loss_func = BCEWithLogitsLoss()
        loss = loss_func(logits.view(-1,num_labels),labels.type_as(logits).view(-1,num_labels))

        return (loss, outputs) if return_outputs else loss

model = model.to(device)

trainer = CustomTrainer(
    model,
    args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("classification_model")
