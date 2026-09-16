"""
Module03 : Embeddings: How Meaning Becomes Numbers
Live-demo script.

Run:
    python part_a_llm_fundamentals/module03_embeddings.py

Requires an embedding model deployment (e.g. text-embedding-3-small)
configured in your .env file.

What this shows:
  1. Turning words into embedding vectors (long lists of numbers that
     represent meaning).
  2. Measuring "closeness in meaning" between two words with cosine similarity
     (a number from -1 to 1 -- closer to 1 means "more similar in meaning").
  3. That semantically related pairs (king/queen) score higher than unrelated
     pairs (king/apple) -- run against BOTH Azure OpenAI and plain OpenAI.
"""
import sys
import os

# Same trick as every other script here -- lets Python find the "common"
# folder at the repo root, since this script lives one folder deeper.
# See module02_tokenization.py for a full line-by-line explanation.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
# numpy gives us fast math on lists of numbers ("vectors") -- we use it
# below to compute cosine similarity between two embeddings.

from common.client import (
    get_azure_client, get_azure_embedding_deployment,
    get_openai_client, get_openai_embedding_model,
)

# The words we'll turn into embeddings. Chosen deliberately: king/queen and
# apple/banana are meaning-related pairs; king/apple and apple/car are not.
WORDS = ["king", "queen", "man", "woman", "apple", "banana", "car"]

# Pairs to compare, with our prediction of what should happen written next
# to each one -- good to read out loud to the class BEFORE running the code.
PAIRS = [
    ("king", "queen"),    # expect: high similarity -- related concepts
    ("king", "man"),      # expect: high similarity
    ("king", "apple"),    # expect: low similarity -- unrelated
    ("apple", "banana"),  # expect: high similarity -- both fruits
    ("apple", "car"),     # expect: low similarity -- unrelated
]


def cosine_similarity(a, b):
    """
    Cosine similarity measures how "aligned" two vectors are, regardless of
    their length -- a common way to compare embeddings. The formula is:
    (a dot b) / (length of a * length of b). You don't need to memorize the
    math -- just know the RESULT: 1.0 means "identical direction / meaning",
    0 means "unrelated", and negative values mean "opposite".
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_demo(label, client, model):
    """
    Does the actual work: ask the model for an embedding (a vector of
    numbers) for every word in WORDS, then compare each pair with cosine
    similarity. Shared by both Option A and Option B below so we don't
    write this logic twice.
    """
    print(f"=== {label} ===\n")
    print("Requesting embeddings for:", WORDS, "\n")

    # One API call gets back one embedding vector PER word in WORDS.
    response = client.embeddings.create(model=model, input=WORDS)

    # zip() pairs up each word with its matching embedding, in order, so we
    # can look up "give me the vector for the word king" easily below.
    vectors = {word: item.embedding for word, item in zip(WORDS, response.data)}

    print(f"Each embedding is a vector of {len(next(iter(vectors.values())))} numbers.\n")

    print("Cosine similarity between pairs:")
    for w1, w2 in PAIRS:
        sim = cosine_similarity(vectors[w1], vectors[w2])
        print(f"  {w1:8s} vs {w2:8s} -> {sim:.4f}")
    print()


def main_azure():
    """Option A: run the embeddings demo against Azure OpenAI."""
    run_demo("Option A: Azure OpenAI embeddings", get_azure_client(), get_azure_embedding_deployment())


def main_openai():
    """Option B: run the exact same demo against the plain OpenAI API."""
    run_demo("Option B: OpenAI API embeddings", get_openai_client(), get_openai_embedding_model())


if __name__ == "__main__":
    main_azure()
    main_openai()
    print("Live-demo idea: swap in your own word pairs, have the class predict the")
    print("score before running, then check how close their intuition was.")
