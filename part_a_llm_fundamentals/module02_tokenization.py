"""
Module 02 : Tokens and Tokenization: How Text Becomes Numbers


Run:
    python part_a_llm_fundamentals/module02_tokenization.py

What this shows:
  1. How a sentence is split into tokens, using the same tokenizer family
     GPT-4-class models use.
  2. That tokens are often smaller than whole words (see the long word below).
  3. That the token count the Azure OpenAI API actually bills you for
     matches what the local tokenizer predicts.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from common.client import get_client, get_chat_deployment

SENTENCES = [
    "I love pizza!",
    "Tokenization",
    "supercalifragilisticexpialidocious",
    "ChatGPT and GPT-4o are OpenAI models.",
]


def show_local_tokenization():
    enc = tiktoken.get_encoding("cl100k_base")  # same family used by GPT-4-class models
    print("=== Local tokenization (no API call needed) ===\n")
    for sentence in SENTENCES:
        token_ids = enc.encode(sentence)
        token_strings = [enc.decode([t]) for t in token_ids]
        print(f'Text: "{sentence}"')
        print(f"  Token count: {len(token_ids)}")
        print(f"  Tokens: {token_strings}")
        print(f"  Token IDs: {token_ids}\n")


def show_api_usage_matches():
    print("=== Confirming against a real Azure OpenAI call ===\n")
    client = get_client()
    deployment = get_chat_deployment()
    text = "I love pizza!"
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": text}],
        max_tokens=1,  # we only care about the prompt token count here
    )
    print(f'Prompt sent: "{text}"')
    print(f"API-reported prompt tokens: {response.usage.prompt_tokens}")
    print("(Compare this to the local tiktoken count above for the same sentence --")
    print(" they should match, since Azure OpenAI bills on the same tokenization.)")


if __name__ == "__main__":
    show_local_tokenization()
    show_api_usage_matches()
