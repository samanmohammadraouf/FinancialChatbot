import os
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import Dataset
import evaluate
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels):

    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = []

    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            weight = 1.0
        else:
            weight = total / (count * num_classes)
            weight = max(weight, 1.0)
        class_weights.append(weight)

    return torch.tensor(class_weights, dtype=torch.float)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
   
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average="micro"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
    }


def train_intent_detection(
    train_df,
    val_df,
    model_name="HooshvareLab/bert-base-parsbert-uncased",
    output_dir="./nlu_model/intent_detection_module",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    early_stopping_patience=2,
    max_length=128,
):

    if not {"text", "label"}.issubset(train_df.columns) or not {"text", "label"}.issubset(val_df.columns):
        raise ValueError("Training and validation files must contain 'text' and 'label' columns.")

    # Label Encoding
    print("Encoding labels...")
    unique_labels = sorted(train_df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"] = val_df["label"].map(label2id)

    # Check Dataset Balance
    print("Checking dataset balance...")
    label_counts = train_df["label_id"].value_counts().sort_index()
    for label_id, count in label_counts.items():
        print(f"Label '{id2label[label_id]}' (ID: {label_id}) has {count} samples.")

    # Compute Class Weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_df["label_id"].tolist())
    print(f"Class Weights: {class_weights}")

    # 4. Tokenizer and Model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing data...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset = val_dataset.rename_column("label_id", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to=None
    )

    # trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience)]
    )

    def compute_loss(model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights.to(device))
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss

    print("Starting training...")
    trainer.train()

    # eval
    print("Evaluating model on training data...")
    train_metrics = trainer.evaluate(train_dataset)
    print("Training Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nEvaluating model on validation data...")
    val_metrics = trainer.evaluate(val_dataset)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")


def predict_intent(text, model_dir="./nlu_model/intent_detection_module"):
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' does not exist. Please train the model first.")

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_label = id2label[predicted_class_id]
    print(f"Input Text: {text}")
    print(f"Predicted Intent: {predicted_label}")
    return predicted_label
