"""
Teaching Guide Module 3 -- Embeddings: How Meaning Becomes Numbers
Live-demo script.

Run:
    python part_a_llm_fundamentals/module03_embeddings.py

Requires an embedding model deployment (e.g. text-embedding-3-small)
configured as AZURE_OPENAI_EMBEDDING_DEPLOYMENT in your .env file.

What this shows:
  1. Turning words into embedding vectors.
  2. Measuring "closeness in meaning" with cosine similarity.
  3. That semantically related pairs score higher than unrelated pairs --
     the same king/queen/apple idea from the Teaching Guide, now with real numbers.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.client import get_client, get_embedding_deployment

WORDS = ["king", "queen", "man", "woman", "apple", "banana", "car"]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    client = get_client()
    deployment = get_embedding_deployment()

    print("Requesting embeddings for:", WORDS, "\n")
    response = client.embeddings.create(model=deployment, input=WORDS)
    vectors = {word: item.embedding for word, item in zip(WORDS, response.data)}

    print(f"Each embedding is a vector of {len(next(iter(vectors.values())))} numbers.\n")

    pairs = [
        ("king", "queen"),    # expect: high similarity -- related concepts
        ("king", "man"),      # expect: high similarity
        ("king", "apple"),    # expect: low similarity -- unrelated
        ("apple", "banana"),  # expect: high similarity -- both fruits
        ("apple", "car"),     # expect: low similarity -- unrelated
    ]

    print("=== Cosine similarity between pairs ===")
    for w1, w2 in pairs:
        sim = cosine_similarity(vectors[w1], vectors[w2])
        print(f"  {w1:8s} vs {w2:8s} -> {sim:.4f}")

    print("\nLive-demo idea: swap in your own word pairs, have the class predict the")
    print("score before running, then check how close their intuition was.")


if __name__ == "__main__":
    main()
