# slot_filling.py

import os
import pandas as pd
import torch
import numpy as np
from typing import Dict, List
from transformers.trainer import Trainer
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
import evaluate

try:
    from seqeval.metrics import precision_score, recall_score, f1_score
except ImportError:
    print("[Warning] seqeval is not installed. Install via pip install seqeval for token-level metrics.")


class IntentSlotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, intents):
        self.encodings = encodings
        self.labels = labels
        self.intents = intents

    def __getitem__(self, idx):
        # Convert dict values to tensors
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["intent_id"] = self.intents[idx]
        return item

    def __len__(self):
        return len(self.labels)


def build_label2id_id2label(all_possible_labels: List[str]):
    label2id = {label: i for i, label in enumerate(sorted(all_possible_labels))}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def build_intent_slot_ids(
    label_slot_dict: Dict[str, List[str]],
    label2id: Dict[str, int]
):
    intent_label_ids = {}
    for intent, slot_list in label_slot_dict.items():
        # all 'b-' and 'i-' variants plus 'o'
        valid_label_strs = set(["o"])
        for slot_base in slot_list:
            valid_label_strs.add(f"b-{slot_base}")
            valid_label_strs.add(f"i-{slot_base}")

        valid_ids = set()
        for lbl_str in valid_label_strs:
            if lbl_str in label2id:
                valid_ids.add(label2id[lbl_str])
        intent_label_ids[intent] = valid_ids
    return intent_label_ids


def parse_slot_string(slot_str):
    return eval(slot_str)

counter_of_outer_range_error = 0
def align_subwords(
    texts: List[str],
    slot_strings: List[str],
    intents: List[str],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 128
):

    all_words = []
    all_slots = []
    for txt, slot_str in zip(texts, slot_strings):
        words = txt.strip().split()
        slot_list = parse_slot_string(slot_str)
        all_words.append(words)
        all_slots.append(slot_list)

    tokenized = tokenizer(
        all_words,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    global counter_of_outer_range_error 
    all_labels = []
    for i, slot_list in enumerate(all_slots):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                # First subword of a word
                if word_idx < len(slot_list):
                    label_str = slot_list[word_idx]
                    label_ids.append(label2id[label_str])
                else:
                    # Out-of-range: default to -100
                    counter_of_outer_range_error += 1
                    label_ids.append(-100)
            else:
                # Subsequent subwords
                label_ids.append(-100)
            prev_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_encodings = {
        k: v.tolist() for k, v in tokenized.items() if k != "overflow_to_sample_mapping"
    }

    return tokenized_encodings, all_labels, intents


class IntentFilteredTrainer(Trainer):
    # this trainer masks invalid classes for each example ... 
    def __init__(self, intent_label_ids: Dict[str, set], id2label: Dict[int, str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_label_ids = intent_label_ids
        self.id2label = id2label

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # extract 'intent_id' from inputs
        intent_id_list = inputs.pop("intent_id")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        batch_size, seq_len, num_labels = logits.shape

        # We'll mask out invalid classes for each sample
        for i in range(batch_size):
            intent_str = intent_id_list[i]
            valid_label_id_set = self.intent_label_ids.get(intent_str, set())
            if not valid_label_id_set:
                continue

            for k in range(num_labels):
                if k not in valid_label_id_set:
                    logits[i, :, k] = -1e9

        loss = None
        loss = F.cross_entropy(
            logits.view(-1, num_labels),
            labels.view(-1),
            ignore_index=-100
        )

        if return_outputs:
            return (loss, outputs)
        return loss

def compute_seqeval_and_rouge(eval_pred, id2label):

    preds, labels = eval_pred
    
    # Convert each to label strings, ignoring subwords where label == -100
    pred_label_list = []
    true_label_list = []
    for p_seq, l_seq in zip(preds, labels):
        p_str_seq = []
        l_str_seq = []
        for p_id, l_id in zip(p_seq, l_seq):
            if l_id == -100:
                continue
            p_str_seq.append(id2label[p_id])
            l_str_seq.append(id2label[l_id])
        pred_label_list.append(p_str_seq)
        true_label_list.append(l_str_seq)

    # seqeval metrics
    try:
        precision = precision_score(true_label_list, pred_label_list)
        recall = recall_score(true_label_list, pred_label_list)
        f1 = f1_score(true_label_list, pred_label_list)
    except:
        precision, recall, f1 = 0.0, 0.0, 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Custome data collator ...
def custom_intent_data_collator(tokenizer):
    base_collator = DataCollatorForTokenClassification(tokenizer)

    def collate_fn(features):
        intent_ids = [f.pop("intent_id") for f in features]

        batch = base_collator(features)
        batch["intent_id"] = intent_ids

        return batch

    return collate_fn

def train_slot_filling(
    train_df,
    val_df,
    label_slot_dict: Dict[str, List[str]],
    model_name: str = "HooshvareLab/bert-base-parsbert-uncased",
    output_dir: str = "./nlu_model/slot_filling_model",
    num_train_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 128
):

    train_texts = train_df["text"].tolist()
    train_intents = train_df["label"].tolist()
    train_slots = train_df["slots"].tolist()

    val_texts = val_df["text"].tolist()
    val_intents = val_df["label"].tolist()
    val_slots = val_df["slots"].tolist()

    # Collect all possible slot labels
    all_slot_tags = set()
    for s in list(train_slots) + list(val_slots):
        slot_list = parse_slot_string(s)
        for slot_tag in slot_list:
            all_slot_tags.add(slot_tag)
    all_slot_tags = sorted(all_slot_tags)

    # Build label2id, id2label
    label2id, id2label = build_label2id_id2label(all_slot_tags)

    # build the per-intent sets of valid label IDs
    intent_label_ids = build_intent_slot_ids(label_slot_dict, label2id)

    # Load tokenizer & model
    print("[SlotFilling] Loading tokenizer/model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    # Tokenize & align
    train_encodings, train_label_ids, train_intents = align_subwords(
        train_texts, train_slots, train_intents, tokenizer, label2id, max_length
    )
    val_encodings, val_label_ids, val_intents = align_subwords(
        val_texts, val_slots, val_intents, tokenizer, label2id, max_length
    )

    train_dataset = IntentSlotDataset(train_encodings, train_label_ids, train_intents)
    val_dataset = IntentSlotDataset(val_encodings, val_label_ids, val_intents)

    data_collator = custom_intent_data_collator(tokenizer)

    # trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        report_to=None,
        remove_unused_columns=False
    )

    def compute_metrics_fn(eval_pred):
        logits, gold_labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_seqeval_and_rouge((preds, gold_labels), id2label)

    trainer = IntentFilteredTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        intent_label_ids=intent_label_ids,
        id2label=id2label,
    )

    print(f'Number of outer index errors: {counter_of_outer_range_error}')

    print("[SlotFilling] Starting training with intent-based slot masking ...")
    trainer.train()

    # Evaluate on train & val sets
    print("[SlotFilling] Evaluating on train set...")
    train_eval = trainer.predict(train_dataset)
    train_metrics = compute_seqeval_and_rouge((np.argmax(train_eval.predictions, axis=-1),
                                              train_eval.label_ids), id2label)
    print("Train metrics:", train_metrics)

    print("[SlotFilling] Evaluating on validation set...")
    val_eval = trainer.predict(val_dataset)
    val_metrics = compute_seqeval_and_rouge((np.argmax(val_eval.predictions, axis=-1),
                                            val_eval.label_ids), id2label)
    print("Validation metrics:", val_metrics)

    print("[SlotFilling] Saving final model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[SlotFilling] Training complete.")


def predict_slots(
    text: str,
    intent: str,
    possible_slots: List[str],
    model_dir: str = "./nlu_model/slot_filling_model",
    max_length: int = 128
):
    """
    Predict slot tags for a given text + intent.
    We mask out invalid classes in logits before argmax.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[SlotFilling] Model directory '{model_dir}' not found. Please train first.")

    # model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    id2label = model.config.id2label

    # Build the full set of label2id
    label2id = {v: k for k, v in id2label.items()}

    # Build a set of valid label IDs for the possible slots
    valid_label_strs = set(["o"])
    for slot_base in possible_slots:
        valid_label_strs.add(f"b-{slot_base}")
        valid_label_strs.add(f"i-{slot_base}")
    valid_label_ids = set()
    for lbl_str in valid_label_strs:
        if lbl_str in label2id:
            valid_label_ids.add(label2id[lbl_str])

    # Prepare input
    words = text.strip().split()
    tokenized = tokenizer(
        [words],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**tokenized)
    logits = outputs.logits 
    logits = logits.squeeze(0) 
    seq_len, num_labels = logits.shape

    # Mask out invalid classes
    for k in range(num_labels):
        if k not in valid_label_ids:
            logits[:, k] = -1e9

    pred_ids = torch.argmax(logits, dim=-1).tolist()

    # Map back to words
    word_ids = tokenized.word_ids(batch_index=0)
    results = []
    prev_word_idx = None
    for i, widx in enumerate(word_ids):
        if widx is None:
            continue
        if widx != prev_word_idx:
            lbl_id = pred_ids[i]
            slot_label = id2label[lbl_id]
            results.append((words[widx], slot_label))
        prev_word_idx = widx

    print(f'slot filling tag results: {results}')
    return results
