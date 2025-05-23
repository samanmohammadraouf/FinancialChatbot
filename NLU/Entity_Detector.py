from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def train_entity_detector(
    train_df,
    val_df,
    model_name="HooshvareLab/bert-base-parsbert-uncased",
    output_dir="./nlu_model/entity_detector/",
    num_train_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=300,
):
    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "O", 1: "E"},
        label2id={"O": 0, "E": 1}
    )

    # Convert labels to numerical values
    def convert_labels(examples):
        examples["labels"] = [[1 if lbl == "E" else 0 for lbl in labels] 
                             for labels in examples["labels"]]
        return examples

    # Prepare datasets
    train_dataset = Dataset.from_dict({
        "tokens": train_df["tokens"].tolist(),
        "labels": [[1 if lbl == "E" else 0 for lbl in label_list] 
                  for label_list in train_df["labels"]]
    })

    val_dataset = Dataset.from_dict({
        "tokens": val_df["tokens"].tolist(),
        "labels": [[1 if lbl == "E" else 0 for lbl in label_list] 
                  for label_list in val_df["labels"]]
    })

    # Tokenization and label alignment
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length"
        )

        labels = []
        for batch_idx in range(len(examples["tokens"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
            label_sequence = examples["labels"][batch_idx]
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # first subword 
                    if word_idx < len(label_sequence):
                        label_ids.append(label_sequence[word_idx])
                    else:
                        label_ids.append(-100)
                else:
                    # subsequent subwords
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # metrics calculation
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_preds = sum(true_predictions, [])
        flat_labels = sum(true_labels, [])

        return {
            "precision": precision_score(flat_labels, flat_preds),
            "recall": recall_score(flat_labels, flat_preds),
            "f1": f1_score(flat_labels, flat_preds),
            "accuracy": accuracy_score(flat_labels, flat_preds),
        }

    # train ...
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def predict_entities(text, model_dir="./nlu_model/entity_detector/", max_length=300):

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Tokenize input
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    word_ids = encoding.word_ids()

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

    # Process results
    results = []
    current_word = None
    current_start = None
    current_end = None
    current_label = None

    for idx, (word_id, pred, offset) in enumerate(zip(word_ids, predictions, offset_mapping)):
        if word_id is None:
            continue 

        if word_id != current_word:
            if current_word is not None:
                results.append((
                    text[current_start:current_end],
                    "E" if current_label == 1 else "O"
                ))
            current_word = word_id
            current_start, current_end = offset
            current_label = pred
        else:
            current_end = offset[1]

    # Add last word
    if current_word is not None:
        results.append((
            text[current_start:current_end],
            "E" if current_label == 1 else "O"
        ))

    return results