import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, TrainingArguments, Trainer, EvalPrediction, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, recall_score, confusion_matrix, precision_score
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
import warnings
import os
warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {device}")
torch.cuda.empty_cache()
print("Cache emptied")

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=3)
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
csv_file = "WGAN-GP.csv"
data = csv_to_dict(csv_file)
for item in data:
    # Convert the string representation of labels into a list
    labels = eval(item.pop('labels'))
    # Add separate keys for each label with their corresponding boolean values
    for label in ['SNV', 'INDEL', 'DUP']:
        item[label] = label in labels


# Load dataset
dataset = pd.DataFrame(data)

# Split dataset into train (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)  
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

print(train_dataset[0])
print(test_dataset[0])
print(val_dataset[0])

train_dataset = train_dataset.remove_columns(["__index_level_0__"])
val_dataset = val_dataset.remove_columns(["__index_level_0__"])
test_dataset = test_dataset.remove_columns(["__index_level_0__"])

labels = [label for label in train_dataset.features.keys() if label not in ['Sequence']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}



def preprocess_data(examples):
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

# Preprocess data
encoded_train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names).shuffle(seed=42)
encoded_val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names).shuffle(seed=42)
encoded_test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names).shuffle(seed=42)

# Set format
encoded_train_dataset.set_format("torch")
encoded_val_dataset.set_format("torch")
encoded_test_dataset.set_format("torch")

input_ids = encoded_train_dataset['input_ids'][0].unsqueeze(0).to(device)
labels = encoded_train_dataset[0]['labels'].unsqueeze(0).to(device)
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

    metrics = {'f1': f1_micro_average,
               'recall' : reca,
               'accuracy': accuracy,
               'precision':precision}
    return metrics

def compute_metrics(p: EvalPrediction):
    with torch.no_grad():  
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

num_labels=3

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")


        outputs = model(**inputs)
        logits = (outputs.logits).to(device)

        # Convert labels to one-hot encoded format
        num_classes = logits.size(-1)
        batch_size = logits.size(0)

        loss_func = BCEWithLogitsLoss()
        loss = loss_func(logits.view(-1,num_labels),labels.type_as(logits).view(-1,num_labels))

        return (loss, outputs) if return_outputs else loss


# Hyperparameter tuning function using Optuna
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 2e-5)
    batch_size = trial.suggest_int("batch_size", 4, 16, step=8)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)


    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=num_labels)
    model = model.to(device)



    # Training arguments
    args = TrainingArguments(
        "trained_dnabert_final",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        do_train=True,
        do_eval=True,
        logging_steps=100,
        logging_strategy='epoch'
    )


    # Custom Trainer
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Training the model
    trainer.train()

    # Evaluate the model
    test_results = trainer.evaluate(encoded_test_dataset)
    print("Test Set Evaluation:", test_results)

    return test_results["eval_f1"]  # Return the F1 score for optimization

# Run Optuna study for hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print the best trial
print("Best trial:", study.best_trial)

# Now, use the best parameters to train the final model
best_params = study.best_trial.params
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
num_train_epochs = best_params["num_train_epochs"]
weight_decay = best_params["weight_decay"]

print(f"Best Hyperparameters: learning_rate={learning_rate}, batch_size={batch_size}, num_train_epochs={num_train_epochs}, weight_decay={weight_decay}")

# Train the model using best hyperparameters
args = TrainingArguments(
    "trained_nucleotide",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    do_train=True,
    do_eval=True,
    logging_steps=100,
    logging_strategy='epoch'
)


tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )


model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=num_labels)
model = model.to(device)



trainer = CustomTrainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save the final model
trainer.save_model("classification_model_dnabert")



test_results = trainer.evaluate(encoded_test_dataset)
print("Test Set Evaluation:", test_results)
