import json
import os
import time
import numpy as np
from pymilvus import MilvusClient, DataType
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MILVUS_URI,
    EMBEDDING_DIM,
    BATCH_SIZE_INSERT,
    DOCUMENTS_PATH,
    EMBEDDINGS_PATH,
    BENCHMARK_RESULTS_PATH,
    INDEX_PARAMS,
)

NUM_QUERIES = 100
TOP_K = 10


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    texts = [doc["text"] for doc in documents]
    return texts, embeddings


def brute_force_search(embeddings, query_embeddings, top_k):
    """Compute exact nearest neighbors using L2 distance."""
    results = []
    for query in query_embeddings:
        distances = np.sum((embeddings - query) ** 2, axis=1)
        top_indices = np.argsort(distances)[:top_k]
        results.append(set(top_indices.tolist()))
    return results


def create_and_index_collection(client, name, texts, embeddings, index_cfg):
    if client.has_collection(name):
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    client.create_collection(collection_name=name, schema=schema)

    total = len(texts)
    for start in range(0, total, BATCH_SIZE_INSERT):
        end = min(start + BATCH_SIZE_INSERT, total)
        batch = [
            {"id": i, "text": texts[i], "embedding": embeddings[i].tolist()}
            for i in range(start, end)
        ]
        client.insert(name, batch)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=index_cfg["index_type"],
        metric_type=index_cfg["metric_type"],
        params=index_cfg["params"],
    )

    t0 = time.time()
    client.create_index(name, index_params)
    client.load_collection(name)
    build_time = time.time() - t0

    return build_time


def run_searches(client, name, query_embeddings, search_params, metric_type):
    latencies = []
    all_results = []

    for query in query_embeddings:
        t0 = time.time()
        results = client.search(
            collection_name=name,
            data=[query.tolist()],
            limit=TOP_K,
            output_fields=["id"],
            search_params={"metric_type": metric_type, "params": search_params},
        )
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        ids = {hit["id"] for hit in results[0]}
        all_results.append(ids)

    return latencies, all_results


def compute_recall(predicted, ground_truth):
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        if len(gt) == 0:
            continue
        recalls.append(len(pred & gt) / len(gt))
    return np.mean(recalls)


def main():
    texts, embeddings = load_data()
    client = MilvusClient(MILVUS_URI)

    rng = np.random.default_rng(seed=42)
    query_indices = rng.choice(len(embeddings), size=NUM_QUERIES, replace=False)
    query_embeddings = embeddings[query_indices]

    print("Computing brute-force baseline...")
    gt_results = brute_force_search(embeddings, query_embeddings, TOP_K)

    results_table = []

    for algo_name, index_cfg in INDEX_PARAMS.items():
        collection_name = f"squad_bench_{algo_name.lower()}"
        print(f"\n{'='*50}")
        print(f"Benchmarking {algo_name}...")
        print(f"{'='*50}")

        build_time = create_and_index_collection(
            client, collection_name, texts, embeddings, index_cfg
        )
        print(f"Index build time: {build_time:.2f}s")

        latencies, search_results = run_searches(
            client,
            collection_name,
            query_embeddings,
            index_cfg["search_params"],
            index_cfg["metric_type"],
        )

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        recall = compute_recall(search_results, gt_results)

        stats = client.get_collection_stats(collection_name)

        results_table.append({
            "algorithm": algo_name,
            "build_time_s": round(build_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "recall_at_10": round(recall, 4),
            "params": index_cfg["params"],
            "search_params": index_cfg["search_params"],
        })

        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")
        print(f"Recall@10: {recall:.4f}")

        client.drop_collection(collection_name)

    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}\n")

    headers = ["Algorithm", "Build Time (s)", "Avg Latency (ms)", "P99 Latency (ms)", "Recall@10"]
    rows = [
        [r["algorithm"], r["build_time_s"], r["avg_latency_ms"], r["p99_latency_ms"], r["recall_at_10"]]
        for r in results_table
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_PATH), exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nResults saved to {BENCHMARK_RESULTS_PATH}")


if __name__ == "__main__":
    main()