"""
Teaching Guide Module 12 -- Chain-of-Thought Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module12_chain_of_thought.py

What this shows: the same math word-problem, once asking for a direct
answer and once asking the model to "think step by step" first. Run
against Azure OpenAI (Option A) and the plain OpenAI API (Option B).
"""
import sys
import os

# Lets Python find the "common" folder at the repo root -- see
# part_a_llm_fundamentals/module02_tokenization.py for the full explanation
# of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)

PROBLEM = ("A basket has 3 apples. You add 2 more baskets with 4 apples each, "
           "then remove 5 apples total. How many apples are left?")

# Ask for just the final number -- no reasoning shown.
DIRECT = f"{PROBLEM} Answer with just the number."

# Ask the model to reason through it out loud first -- this is the whole
# "chain-of-thought" technique in one sentence: "Think step by step."
COT = f"{PROBLEM} Think step by step, then give the final answer on the last line."


def ask(client, deployment, prompt, max_tokens):
    """Small shared helper: send one prompt, return the model's reply as text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,  # temperature=0 -> most consistent, least random answer
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    print("=== Direct answer (no reasoning shown) ===")
    print(ask(client, deployment, DIRECT, max_tokens=10))  # short cap -- we only expect one number

    print("\n=== Chain-of-thought ('think step by step') ===")
    print(ask(client, deployment, COT, max_tokens=200))    # longer cap -- reasoning takes more words
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())

    print("Run both a few times -- direct answers are more likely to be wrong or")
    print("inconsistent; chain-of-thought tends to be more reliable on multi-step")
    print("arithmetic like this. The correct answer is 6.")
