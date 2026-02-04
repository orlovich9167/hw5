import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MILVUS_URI,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BATCH_SIZE_EMBED,
    BATCH_SIZE_INSERT,
    DOCUMENTS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PARAMS,
)


def load_documents():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_embeddings(texts):
    print(f"Loading model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Computing embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE_EMBED, show_progress_bar=True)
    return np.array(embeddings)


def create_collection(client):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    hnsw_params = INDEX_PARAMS["HNSW"]
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=hnsw_params["index_type"],
        metric_type=hnsw_params["metric_type"],
        params=hnsw_params["params"],
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}' with HNSW index")


def insert_data(client, texts, embeddings):
    total = len(texts)
    for start in range(0, total, BATCH_SIZE_INSERT):
        end = min(start + BATCH_SIZE_INSERT, total)
        batch = [
            {"text": texts[i], "embedding": embeddings[i].tolist()}
            for i in range(start, end)
        ]
        client.insert(COLLECTION_NAME, batch)
        print(f"Inserted {end}/{total}")


def main():
    documents = load_documents()
    texts = [doc["text"] for doc in documents]

    embeddings = compute_embeddings(texts)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Cached embeddings to {EMBEDDINGS_PATH}")

    client = MilvusClient(MILVUS_URI)
    create_collection(client)
    insert_data(client, texts, embeddings)

    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"Collection stats: {stats}")


if __name__ == "__main__":
    main()