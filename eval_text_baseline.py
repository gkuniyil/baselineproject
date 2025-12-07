"""
Evaluate the unsupervised text retrieval baseline.

Code does the following:
    1. Load the FAISS index and metadata from disk.
    2. Load the MiniLM text encoder used to build the index.
    3. Provide a helper function `search(query, top_k)` that returns
       the top-k (idx, uid, caption, score) for a given text query.
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


# configuration
FAISS_INDEX_PATH = "data/index/text_index.faiss"
METADATA_PATH = "data/index/text_index_metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TESTS_JSON_PATH = "tests/baseline_robustness_tests.json"
TOP_K_DEFAULT = 10


def load_resources():
    """
    load FAISS index, metadata DataFrame, and MiniLM model.

    returns:
        index: FAISS index
        meta_df: DataFrame with columns ['uid', 'caption', 'image_path']
        model: SentenceTransformer model for query encoding
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_PATH}")

    print("[INFO] Loading FAISS index from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)

    print("[INFO] Loading metadata from disk...")
    meta_df = pd.read_csv(METADATA_PATH)
    print(f"[INFO] Loaded metadata with {len(meta_df)} rows.")

    print(f"[INFO] Loading MiniLM model ({MODEL_NAME}) for query encoding...")
    model = SentenceTransformer(MODEL_NAME)

    return index, meta_df, model


def encode_and_normalize_query(model, query: str) -> np.ndarray:
    """
    encode a single query string with MiniLM and L2-normalize the embedding

    input:
        model: SentenceTransformer
        query: text string

    output:
        q_emb_norm: NumPy array of shape (1, D), dtype float32
    """
    q_emb = model.encode([query], show_progress_bar=False)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
    q_emb_norm = (q_emb / norms).astype("float32")
    return q_emb_norm


def cosine_similarity(model, q1: str, q2: str) -> float:
    """
    compute cosine similarity between two query strings
    using the same MiniLM encoder and L2-normalization.
    """
    emb1 = encode_and_normalize_query(model, q1)  # shape (1, D)
    emb2 = encode_and_normalize_query(model, q2)  # shape (1, D)

    # since they are normalized, cosine = dot product
    sim = float(np.dot(emb1[0], emb2[0]))
    return sim


def search(index, meta_df, model, query: str, top_k: int = TOP_K_DEFAULT):
    """
    run a text query against the FAISS index and return top k results

    inputs:
        index: FAISS index (IndexFlatIP with normalized embeddings)
        meta_df: DataFrame with metadata (uid, caption, image_path)
        model: MiniLM SentenceTransformer
        query: text query string
        top_k: number of results to return

    output:
        results: list of dicts like:
            {
                "rank": int,
                "idx": int,
                "uid": str,
                "caption": str,
                "score": float,
            }
    """
    q_emb_norm = encode_and_normalize_query(model, query)
    scores, indices = index.search(q_emb_norm, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        row = meta_df.iloc[idx]
        results.append(
            {
                "rank": rank,
                "idx": int(idx),
                "uid": row["uid"],
                "caption": row["caption"],
                "score": float(score),
            }
        )
    return results


def evaluate_random_subset(
    index,
    meta_df,
    model,
    num_samples: int = 100,
    top_k: int = 10,
    seed: int = 42,
):
    """
    evaluate retrieval performance on a random subset of captions.

    for each sampled row:
        - use its caption as the query
        - treat its UID as the "correct" asset
        - run search(top_k)
        - compute R@1, R@5, R@10 and Reciprocal Rank

    returns:
        dict with averaged metrics: R@1, R@5, R@10, MRR, num_samples
    """
    rng = np.random.default_rng(seed)
    n = len(meta_df)
    if num_samples > n:
        num_samples = n

    sampled_indices = rng.choice(n, size=num_samples, replace=False)

    r_at_1 = 0
    r_at_5 = 0
    r_at_10 = 0
    mrr_sum = 0.0

    for count, idx in enumerate(sampled_indices, start=1):
        row = meta_df.iloc[idx]
        true_uid = row["uid"]
        query_caption = row["caption"]

        results = search(index, meta_df, model, query_caption, top_k=top_k)

        # find rank of the correct UID in the results
        rank_of_true = None
        for r in results:
            if r["uid"] == true_uid:
                rank_of_true = r["rank"]
                break

        if rank_of_true is not None:
            if rank_of_true <= 1:
                r_at_1 += 1
            if rank_of_true <= 5:
                r_at_5 += 1
            if rank_of_true <= 10:
                r_at_10 += 1

            # Reciprocal Rank
            mrr_sum += 1.0 / rank_of_true

        if count % 10 == 0:
            print(f"[INFO] Processed {count}/{num_samples} samples...")

    num = float(num_samples)
    metrics = {
        "R@1": r_at_1 / num,
        "R@5": r_at_5 / num,
        "R@10": r_at_10 / num,
        "MRR": mrr_sum / num,
        "num_samples": num_samples,
    }
    return metrics


def run_robustness_tests(index, meta_df, model, tests_path, top_ks=(1, 5, 10)):
    """
    Run robustness tests over a set of query families.

    tests: list of dicts, each like:
        {
            "name": str,
            "orig": str,
            "orig_type": str (optional, default "canonical"),
            "variants": [
                {"query": str, "type": str},
                ...
            ]
        }

    For each test:
      1. Use original query to define target UID (top-1 result).
      2. For original + each variant:
         - Run search with top_k = max(top_ks).
         - Find rank of target UID if present.
         - Classify as R@1, R@5, R@10, or miss.

    Returns:
      results: list of dicts with keys:
        - test_name
        - query_type
        - variant_label
        - variant_type
        - query
        - rank
        - success_level
    """
    
    with open(tests_path, "r", encoding="utf-8") as f:
        tests = json.load(f)


    max_k = max(top_ks)
    top_ks_sorted = sorted(top_ks)

    def classify_rank(rank):
        if rank is None:
            return "miss"
        for k in top_ks_sorted:
            if rank <= k:
                return f"R@{k}"
        return "miss"

    results = []

    def evaluate_query(query, query_type, variant_label, variant_type, target_uid, test_name):
        res = search(index, meta_df, model, query, top_k=max_k)

        rank = None
        for r in res:
            if r["uid"] == target_uid:
                rank = r["rank"]
                break

        success = classify_rank(rank)

        results.append(
            {
                "test_name": test_name,
                "query_type": query_type,        # "orig" or "variant"
                "variant_label": variant_label,  # "orig" or the variant query
                "variant_type": variant_type,    # e.g. "typo", "hypernym", etc.
                "query": query,
                "rank": rank,
                "success_level": success,
            }
        )

        return rank, success

    print("[INFO] Running robustness tests (R@1 / R@5 / R@10)...")

    for test in tests:
        name = test.get("name", "UNKNOWN_TEST")
        orig_query = test["orig"]
        orig_type = test.get("orig_type", "canonical")
        variants = test.get("variants", [])

        print("\n" + "-" * 80+"\n")
        print(f"[TEST] {name}")
        print(f"  Original query: {orig_query}")

        # determine target UID from original query (top-1 result)
        orig_results = search(index, meta_df, model, orig_query, top_k=max_k)
        orig_top = orig_results[0]
        target_uid = orig_top["uid"]
        print(
            "  [ORIG TOP-1] uid={}  score={:.4f}".format(
                target_uid, orig_top["score"]
            )
        )
        print("               caption: {}".format(orig_top["caption"]))

        # evaluate original query
        orig_rank, orig_level = evaluate_query(
            query=orig_query,
            query_type="orig",
            variant_label="orig",
            variant_type=orig_type,
            target_uid=target_uid,
            test_name=name,
        )
        print(f"\n  => Original: success={orig_level}  rank={orig_rank}  type={orig_type}")

        # pretty print variants in a table-like format
        if variants:
            print("\n  Variants:")
            # header
            print("    {succ:<7}  {rank:<4}  {vtype:<20}  {query}".format(
                succ="success",
                rank="rank",
                vtype="type",
                query="query",
            ))
            print("    {:-<7}  {:-<4}  {:-<20}  {:-<40}".format("", "", "", ""))

            for v in variants:
                q = v["query"]
                v_type = v.get("type", "unknown")

                rank_v, level_v = evaluate_query(
                    query=q,
                    query_type="variant",
                    variant_label=q,
                    variant_type=v_type,
                    target_uid=target_uid,
                    test_name=name,
                )

                rank_str = "-" if rank_v is None else str(rank_v)
                print(
                    "    {succ:<7}  {rank:<4}  {vtype:<20}  {query}".format(
                        succ=level_v,
                        rank=rank_str,
                        vtype=v_type[:20],
                        query=q,
                    )
                )

    return results

def main():
    # load index, metadata, and text encoder
    index, meta_df, model = load_resources()

    # quick sanity-check queries
    test_queries = [
        "white sofa with wooden legs",
        "airplane",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"[QUERY] {q}")
        results = search(index, meta_df, model, q, top_k=5)
        for r in results:
            print(f"  {r['rank']}. uid={r['uid']}  score={r['score']:.4f}")
            print(f"     caption: {r['caption']}")

    # evaluation on a random subset
    print("\n" + "=" * 80)
    print("[INFO] Starting evaluation on random subset of captions...")

    metrics = evaluate_random_subset(
        index,
        meta_df,
        model,
        num_samples=100,
        top_k=10,
    )

    print("\nEvaluation Summary ({} samples):".format(metrics["num_samples"]))
    print("  R@1  = {:.3f}".format(metrics["R@1"]))
    print("  R@5  = {:.3f}".format(metrics["R@5"]))
    print("  R@10 = {:.3f}".format(metrics["R@10"]))
    print("  MRR  = {:.3f}".format(metrics["MRR"]))

    # small robustness check
    run_robustness_tests(index, meta_df, model, TESTS_JSON_PATH)

if __name__ == "__main__":
    main()
