"""
Evaluate the unsupervised text retrieval baseline.

Code does the following:
    1. Load the FAISS index and metadata from disk.
    2. Load the MiniLM text encoder used to build the index.
    3. Provide a helper function `search(query, top_k)` that returns
       the top-k (idx, uid, caption, score) for a given text query.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


# configuration
FAISS_INDEX_PATH = "data/index/text_index.faiss"
METADATA_PATH = "data/index/text_index_metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
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


def run_robustness_tests(index, meta_df, model):
    """
    run a small robustness test with:
        - synonym substitutions
        - added adjectives
        - word reordering
    (single concept: sofa)
    """
    tests = [
        {
            "name": "sofa",
            "orig": "white sofa with wooden legs",
            "variants": [
                "white couch with wooden legs",            # synonym
                "comfortable white sofa with wooden legs", # added adjective
                "sofa with wooden legs that is white",     # changed order
            ],
        },
    ]

    print("\n" + "=" * 80)
    print("[INFO] Running robustness test (synonyms / wording changes)...")

    for test in tests:
        orig = test["orig"]
        variants = test["variants"]

        print("\n" + "-" * 80)
        print(f"  Original query: {orig}")

        orig_results = search(index, meta_df, model, orig, top_k=5)
        orig_top = orig_results[0]
        orig_uid = orig_top["uid"]
        print(
            "  [ORIG TOP-1] uid={}  score={:.4f}".format(
                orig_uid, orig_top["score"]
            )
        )
        print("               caption: {}".format(orig_top["caption"]))

        for v in variants:
            sim = cosine_similarity(model, orig, v)
            v_results = search(index, meta_df, model, v, top_k=5)
            v_top = v_results[0]
            same_uid = (v_top["uid"] == orig_uid)

            print("\n    Variant query: {}".format(v))
            print("      cosine_sim(orig, variant) = {:.4f}".format(sim))
            print(
                "      [VAR TOP-1] uid={}  score={:.4f}".format(
                    v_top["uid"], v_top["score"]
                )
            )
            print("                  caption: {}".format(v_top["caption"]))
            print("      Top-1 same as original? {}".format(same_uid))


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
    run_robustness_tests(index, meta_df, model)


if __name__ == "__main__":
    main()
