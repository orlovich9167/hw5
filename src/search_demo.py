import json
import os
import sys

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    WeightedRanker,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE_INSERT,
    DOCUMENTS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    MILVUS_URI,
    SAMPLE_QUERIES,
    SEARCH_COLLECTION_NAME,
    SEARCH_RESULTS_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_PATH,
)

TOP_K = 10


def load_data():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    return documents, embeddings


def build_tfidf(texts):
    print("Building TF-IDF model...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = vectorizer.fit_transform(texts)

    tfidf_data = {
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "max_features": TFIDF_MAX_FEATURES,
    }
    with open(TFIDF_PATH, "w") as f:
        json.dump(tfidf_data, f)

    print(f"TF-IDF model saved to {TFIDF_PATH}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer, tfidf_matrix


def sparse_to_milvus(row):
    coo = row.tocoo()
    return {int(idx): float(val) for idx, val in zip(coo.col, coo.data)}


def create_collection(client):
    if client.has_collection(SEARCH_COLLECTION_NAME):
        client.drop_collection(SEARCH_COLLECTION_NAME)
        print(f"Dropped existing collection '{SEARCH_COLLECTION_NAME}'")

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("text_length", DataType.INT64)
    schema.add_field("sentence_count", DataType.INT64)
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_embedding",
        index_type="HNSW",
        metric_type="L2",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )

    client.create_collection(
        collection_name=SEARCH_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{SEARCH_COLLECTION_NAME}'")


def insert_data(client, documents, embeddings, tfidf_matrix):
    total = len(documents)
    for start in range(0, total, BATCH_SIZE_INSERT):
        end = min(start + BATCH_SIZE_INSERT, total)
        batch = []
        for i in range(start, end):
            doc = documents[i]
            batch.append({
                "id": doc["id"],
                "text": doc["text"],
                "title": doc["title"],
                "text_length": doc["text_length"],
                "sentence_count": doc["sentence_count"],
                "dense_embedding": embeddings[i].tolist(),
                "sparse_embedding": sparse_to_milvus(tfidf_matrix[i]),
            })
        client.insert(SEARCH_COLLECTION_NAME, batch)
        print(f"Inserted {end}/{total}")


def print_results(results, title_header="Results"):
    print(f"\n  {title_header}:")
    rows = []
    for rank, hit in enumerate(results, 1):
        entity = hit.get("entity", {})
        text_preview = entity.get("text", "")[:100] + "..."
        rows.append([
            rank,
            f"{hit['distance']:.4f}",
            entity.get("title", "N/A"),
            text_preview,
        ])
    print(tabulate(rows, headers=["#", "Score", "Title", "Text"], tablefmt="simple"))


def hits_to_list(results):
    out = []
    for rank, hit in enumerate(results, 1):
        entity = hit.get("entity", {})
        out.append({
            "rank": rank,
            "score": round(float(hit["distance"]), 4),
            "title": entity.get("title", "N/A"),
            "text": entity.get("text", "")[:200],
            "text_length": entity.get("text_length"),
        })
    return out


def encode_query_dense(query, model):
    return model.encode([query])[0]


def encode_query_sparse(query, vectorizer):
    sparse_vec = vectorizer.transform([query])
    return sparse_to_milvus(sparse_vec[0])


def scenario_dense_search(client, query, query_embedding):
    print(f"\n{'='*60}")
    print(f"SCENARIO 1: Dense Vector Search (L2)")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    results = client.search(
        collection_name=SEARCH_COLLECTION_NAME,
        data=[query_embedding.tolist()],
        limit=TOP_K,
        output_fields=["title", "text"],
        search_params={"metric_type": "L2", "params": {"ef": 64}},
        anns_field="dense_embedding",
    )
    print_results(results[0])
    return {"scenario": "dense_search", "query": query, "results": hits_to_list(results[0])}


def scenario_sparse_search(client, query, query_sparse):
    print(f"\n{'='*60}")
    print(f"SCENARIO 2: Sparse Search (TF-IDF, IP)")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    results = client.search(
        collection_name=SEARCH_COLLECTION_NAME,
        data=[query_sparse],
        limit=TOP_K,
        output_fields=["title", "text"],
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        anns_field="sparse_embedding",
    )
    print_results(results[0])
    return {"scenario": "sparse_search", "query": query, "results": hits_to_list(results[0])}


def scenario_hybrid_search(client, query, query_embedding, query_sparse):
    print(f"\n{'='*60}")
    print(f"SCENARIO 3: Hybrid Search (Dense + Sparse, WeightedRanker)")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    weight_combos = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    results_info = []

    for dense_w, sparse_w in weight_combos:
        print(f"\n  Weights: dense={dense_w}, sparse={sparse_w}")

        dense_req = AnnSearchRequest(
            data=[query_embedding.tolist()],
            anns_field="dense_embedding",
            param={"metric_type": "L2", "params": {"ef": 64}},
            limit=TOP_K,
        )
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=TOP_K,
        )

        results = client.hybrid_search(
            collection_name=SEARCH_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(dense_w, sparse_w),
            limit=TOP_K,
            output_fields=["title", "text"],
        )
        print_results(results[0], f"Hybrid (dense={dense_w}, sparse={sparse_w})")
        results_info.append({
            "dense_weight": dense_w,
            "sparse_weight": sparse_w,
            "results": hits_to_list(results[0]),
        })

    return {"scenario": "hybrid_search", "query": query, "weight_combos": results_info}


def scenario_metadata_filter(client, query, query_embedding, documents):
    print(f"\n{'='*60}")
    print(f"SCENARIO 4: Metadata Filtering")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    # Filter by title
    titles = list({doc["title"] for doc in documents})
    sample_title = titles[0]
    print(f"\n  Filter: title == \"{sample_title}\"")

    results = client.search(
        collection_name=SEARCH_COLLECTION_NAME,
        data=[query_embedding.tolist()],
        limit=TOP_K,
        output_fields=["title", "text", "text_length"],
        search_params={"metric_type": "L2", "params": {"ef": 64}},
        anns_field="dense_embedding",
        filter=f'title == "{sample_title}"',
    )
    title_results = hits_to_list(results[0])
    print_results(results[0], f"Filtered by title='{sample_title}'")

    # Filter by text_length
    lengths = [doc["text_length"] for doc in documents]
    median_length = int(np.median(lengths))
    print(f"\n  Filter: text_length > {median_length} (median)")

    results = client.search(
        collection_name=SEARCH_COLLECTION_NAME,
        data=[query_embedding.tolist()],
        limit=TOP_K,
        output_fields=["title", "text", "text_length"],
        search_params={"metric_type": "L2", "params": {"ef": 64}},
        anns_field="dense_embedding",
        filter=f"text_length > {median_length}",
    )
    length_results = hits_to_list(results[0])
    print_results(results[0], f"Filtered by text_length > {median_length}")

    return {
        "scenario": "metadata_filter",
        "query": query,
        "title_filter": sample_title,
        "title_filter_results": title_results,
        "length_filter": median_length,
        "length_filter_results": length_results,
    }


def scenario_similarity_metrics(client, query, query_embedding, documents, embeddings):
    print(f"\n{'='*60}")
    print(f"SCENARIO 5: Similarity Metrics Comparison (L2, COSINE, IP)")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    metrics_results = []

    for metric in ["L2", "COSINE", "IP"]:
        temp_name = f"squad_metric_{metric.lower()}"

        if client.has_collection(temp_name):
            client.drop_collection(temp_name)

        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=metric,
            params={"M": 16, "efConstruction": 200},
        )

        client.create_collection(collection_name=temp_name, schema=schema, index_params=index_params)

        total = len(documents)
        for start in range(0, total, BATCH_SIZE_INSERT):
            end = min(start + BATCH_SIZE_INSERT, total)
            batch = [
                {
                    "id": documents[i]["id"],
                    "text": documents[i]["text"],
                    "title": documents[i]["title"],
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(start, end)
            ]
            client.insert(temp_name, batch)

        print(f"\n  Building {metric} index...")
        results = client.search(
            collection_name=temp_name,
            data=[query_embedding.tolist()],
            limit=TOP_K,
            output_fields=["title", "text"],
            search_params={"metric_type": metric, "params": {"ef": 64}},
        )
        print_results(results[0], f"Metric: {metric}")

        metrics_results.append({"metric": metric, "results": hits_to_list(results[0])})

        client.drop_collection(temp_name)

    return {"scenario": "similarity_metrics", "query": query, "metrics": metrics_results}


def scenario_top_k(client, query, query_embedding):
    print(f"\n{'='*60}")
    print(f"SCENARIO 6: Different top-k values")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    k_values = [5, 10, 20]
    k_results = []

    for k in k_values:
        results = client.search(
            collection_name=SEARCH_COLLECTION_NAME,
            data=[query_embedding.tolist()],
            limit=k,
            output_fields=["title", "text"],
            search_params={"metric_type": "L2", "params": {"ef": 64}},
            anns_field="dense_embedding",
        )
        print_results(results[0], f"top-k={k}")
        k_results.append({"k": k, "results": hits_to_list(results[0])})

    return {"scenario": "top_k", "query": query, "k_values": k_results}


def main():
    print("Loading data...")
    documents, embeddings = load_data()
    texts = [doc["text"] for doc in documents]

    vectorizer, tfidf_matrix = build_tfidf(texts)

    print("Connecting to Milvus...")
    client = MilvusClient(MILVUS_URI)

    create_collection(client)
    insert_data(client, documents, embeddings, tfidf_matrix)

    stats = client.get_collection_stats(SEARCH_COLLECTION_NAME)
    print(f"Collection stats: {stats}")

    print("\nLoading embedding model for query encoding...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_results = []

    query = SAMPLE_QUERIES[0]
    query_embedding = encode_query_dense(query, model)
    query_sparse = encode_query_sparse(query, vectorizer)

    # Scenario 1: Dense search
    all_results.append(scenario_dense_search(client, query, query_embedding))

    # Scenario 2: Sparse search
    all_results.append(scenario_sparse_search(client, query, query_sparse))

    # Scenario 3: Hybrid search
    all_results.append(scenario_hybrid_search(client, query, query_embedding, query_sparse))

    # Scenario 4: Metadata filter
    all_results.append(scenario_metadata_filter(client, query, query_embedding, documents))

    # Scenario 5: Similarity metrics (uses different query for variety)
    query2 = SAMPLE_QUERIES[1]
    query_embedding2 = encode_query_dense(query2, model)
    all_results.append(
        scenario_similarity_metrics(client, query2, query_embedding2, documents, embeddings)
    )

    # Scenario 6: Top-k
    query3 = SAMPLE_QUERIES[2]
    query_embedding3 = encode_query_dense(query3, model)
    all_results.append(scenario_top_k(client, query3, query_embedding3))

    # Save results
    os.makedirs(os.path.dirname(SEARCH_RESULTS_PATH), exist_ok=True)
    with open(SEARCH_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to {SEARCH_RESULTS_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()