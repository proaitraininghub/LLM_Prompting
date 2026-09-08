"""
Teaching Guide Module 15 -- Iterative Refinement
Live-demo script.

Run:
    python part_b_prompt_engineering/module15_iterative_refinement.py

What this shows: treating a first response as a draft, then asking the
model to revise it -- twice -- watching the output change each time,
with the full conversation history carried forward each turn.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment


def ask(client, deployment, messages, max_tokens=200):
    response = client.chat.completions.create(
        model=deployment, messages=messages, max_tokens=max_tokens, temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    messages = [{"role": "user", "content": "Write a short product description for a smart water bottle."}]
    print("=== Draft 1 ===")
    draft1 = ask(client, deployment, messages)
    print(draft1)
    messages.append({"role": "assistant", "content": draft1})

    messages.append({"role": "user", "content": "Make it punchier and add one call to action."})
    print("\n=== Draft 2 (refinement 1) ===")
    draft2 = ask(client, deployment, messages)
    print(draft2)
    messages.append({"role": "assistant", "content": draft2})

    messages.append({"role": "user", "content": "Now shorten it to under 30 words."})
    print("\n=== Draft 3 (refinement 2) ===")
    draft3 = ask(client, deployment, messages)
    print(draft3)


if __name__ == "__main__":
    main()
