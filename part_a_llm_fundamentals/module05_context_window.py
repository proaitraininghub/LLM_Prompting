"""
Teaching Guide Module 5 -- The Context Window: The Model's Short-Term Memory
Live-demo script.

Run:
    python part_a_llm_fundamentals/module05_context_window.py

What this shows:
  1. Counting tokens in a growing conversation, turn by turn.
  2. What fraction of the deployed model's context window that represents.
  3. That the model can recall something mentioned early on -- as long as
     it's still inside the context window that gets resent every turn.
  Run against Azure OpenAI (Option A) and the plain OpenAI API (Option B).
"""
import sys
import os

# Lets Python find the "common" folder at the repo root -- see
# module02_tokenization.py for the full explanation of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)

# Update this to match your deployed model's published context window size
# (the maximum number of tokens it can "see" at once, including everything
# sent so far in the conversation).
MODEL_CONTEXT_WINDOW = 128_000


def count_tokens(messages, encoding_name="cl100k_base"):
    """
    Adds up how many tokens the WHOLE conversation-so-far would cost, by
    tokenizing every message and summing the counts (plus a small per-message
    overhead that real APIs also charge for formatting).
    """
    enc = tiktoken.get_encoding(encoding_name)
    total = 0
    for m in messages:
        total += len(enc.encode(m["content"])) + 4  # rough per-message overhead
    return total


def main(label, client, model):
    """
    Simulates a conversation that grows turn by turn: we "tell" the model a
    new fact each time, track how many tokens that's using up out of the
    context window, then finally ask it to recall something from earlier --
    proving the model only "remembers" what's still being resent to it.
    """
    print(f"#### {label} ####\n")

    # "messages" is the full running conversation history. Every single API
    # call resends this ENTIRE list -- the model has no memory of its own
    # between calls; whatever isn't in this list, it simply doesn't know.
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    facts = [
        "My favorite color is teal.",
        "I have a dog named Biscuit.",
        "I live in Pune.",
        "My favorite food is dosa.",
    ]

    print("=== Building up a conversation and tracking token count ===\n")
    for fact in facts:
        messages.append({"role": "user", "content": fact})
        messages.append({"role": "assistant", "content": "Got it, noted!"})
        tokens_so_far = count_tokens(messages)
        pct = tokens_so_far / MODEL_CONTEXT_WINDOW
        print(f"After telling the model '{fact}'")
        print(f"  -> running total: {tokens_so_far} tokens ({pct:.4%} of a {MODEL_CONTEXT_WINDOW:,}-token window)")

    print("\n=== Asking the model to recall something from earlier ===")
    messages.append({"role": "user", "content": "What's my favorite color, and what's my dog's name?"})
    # This call sends the ENTIRE messages list built up above -- that's the
    # only reason the model can answer correctly here.
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=60)
    print("Model answer:", response.choices[0].message.content.strip())
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())

    print("Live-demo idea: wrap the fact-adding loop to append hundreds of filler facts,")
    print("then manually truncate the oldest messages out of `messages` before asking the")
    print("recall question again -- watch the model fail to recall what got truncated away,")
    print("even though it answered correctly a moment ago.")
