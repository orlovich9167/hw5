import json
import os
import re

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DOCUMENTS_PATH, MAX_DOCUMENTS

from datasets import load_dataset


def compute_sentence_count(text):
    parts = re.split(r'[.!?]+', text.strip())
    count = len([p for p in parts if p.strip()])
    return max(count, 1)


def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="train")

    context_to_title = {}
    for row in dataset:
        ctx = row["context"]
        if ctx not in context_to_title:
            context_to_title[ctx] = row["title"]

    print(f"Total unique contexts: {len(context_to_title)}")

    items = list(context_to_title.items())[:MAX_DOCUMENTS]
    print(f"Using {len(items)} documents")

    documents = []
    for i, (text, title) in enumerate(items):
        documents.append({
            "id": i,
            "text": text,
            "title": title,
            "text_length": len(text),
            "sentence_count": compute_sentence_count(text),
        })

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    titles = set(doc["title"] for doc in documents)
    print(f"Saved to {DOCUMENTS_PATH}")
    print(f"Unique titles: {len(titles)}")
    print(f"Sample titles: {list(titles)[:5]}")


if __name__ == "__main__":
    prepare_data()