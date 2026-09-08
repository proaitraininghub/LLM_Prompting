"""
Teaching Guide Module 8 -- A Worked Walkthrough: Tracing One Prompt Start to Finish
Live-demo script.

Run:
    python part_a_llm_fundamentals/module08_full_walkthrough.py

Traces a single prompt through every stage covered in Part A: tokenization,
the model generating a response, temperature's effect on that generation,
and the context window that would hold the exchange -- ending with the
token usage the API actually reports.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from common.client import get_client, get_chat_deployment

PROMPT = "Explain gravity simply."


def main():
    client = get_client()
    deployment = get_chat_deployment()
    enc = tiktoken.get_encoding("cl100k_base")

    print("=== Step 1: Tokenize the prompt ===")
    print(f'Prompt: "{PROMPT}"')
    tokens = enc.encode(PROMPT)
    print(f"Tokens ({len(tokens)}): {[enc.decode([t]) for t in tokens]}\n")

    print("=== Step 2: Send to the model, low temperature (more predictable) ===")
    response_low = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.0,
        max_tokens=80,
    )
    print(response_low.choices[0].message.content.strip())
    u = response_low.usage
    print(f"\nToken usage -- prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}\n")

    print("=== Step 3: Same prompt, high temperature (more varied) ===")
    response_high = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=1.2,
        max_tokens=80,
    )
    print(response_high.choices[0].message.content.strip())

    print("\n=== Step 4: The context window would hold this whole exchange ===")
    print("If a student asked a follow-up like 'Explain it again for a 5-year-old',")
    print("the model would see this entire exchange again as part of its context --")
    print("nothing here is remembered outside of what's resent in the messages list.")


if __name__ == "__main__":
    main()
