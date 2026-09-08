"""
Teaching Guide Module 12 -- Chain-of-Thought Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module12_chain_of_thought.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

PROBLEM = ("A basket has 3 apples. You add 2 more baskets with 4 apples each, "
           "then remove 5 apples total. How many apples are left?")

DIRECT = f"{PROBLEM} Answer with just the number."
COT = f"{PROBLEM} Think step by step, then give the final answer on the last line."


def ask(client, deployment, prompt, max_tokens):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    print("=== Direct answer (no reasoning shown) ===")
    print(ask(client, deployment, DIRECT, max_tokens=10))

    print("\n=== Chain-of-thought ('think step by step') ===")
    print(ask(client, deployment, COT, max_tokens=200))

    print("\nRun both a few times -- direct answers are more likely to be wrong or")
    print("inconsistent; chain-of-thought tends to be more reliable on multi-step")
    print("arithmetic like this. The correct answer is 6.")


if __name__ == "__main__":
    main()
