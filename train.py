# =========================
# 1. Imports
# =========================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, pipeline
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# 2. Load Dataset
# =========================
dataset = load_dataset("imdb")

train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(300))

# =========================
# 3. Tokenizer
# =========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# =========================
# 4. Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

# =========================
# 5. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# =========================
# 6. Training Arguments (SAFE VERSION)
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # smaller for laptop
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
)

# =========================
# 7. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# =========================
# 8. Train
# =========================
print("Training started...")
trainer.train()

# =========================
# 9. Evaluate
# =========================
print("\nEvaluation:")
trainer.evaluate()

# =========================
# 10. Save Model
# =========================
model.save_pretrained("sentiment-model")
tokenizer.save_pretrained("sentiment-model")

# =========================
# 11. Test Model
# =========================
classifier = pipeline("sentiment-analysis", model="sentiment-model")

print("\nTesting Model:")
print(classifier("This internship is amazing!"))
print(classifier("This is the worst experience ever."))