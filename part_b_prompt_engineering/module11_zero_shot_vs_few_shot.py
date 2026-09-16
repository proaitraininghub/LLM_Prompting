"""
Module 11 -- Zero-Shot vs. Few-Shot Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module11_zero_shot_vs_few_shot.py

What this shows: the same classification task, once asked "cold" (zero-shot)
and once after showing the model two worked examples first (few-shot).
Run against Azure OpenAI (Option A) and the plain OpenAI API (Option B).
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

REVIEW_TO_CLASSIFY = "The screen cracked after one week and support never replied."

# Zero-shot: we just ask directly, with no examples of how to do the task.
ZERO_SHOT = f'Classify the sentiment of this review as positive, negative, or neutral: "{REVIEW_TO_CLASSIFY}"'

# Few-shot: we show the model two solved examples FIRST, then give it the
# real review to classify -- this "teaches by example" within the prompt
# itself, without any actual training/fine-tuning of the model.
FEW_SHOT = f"""Classify each review's sentiment as positive, negative, or neutral.

Review: "Fast shipping and works perfectly!"
Sentiment: positive

Review: "It's okay, does the job."
Sentiment: neutral

Review: "{REVIEW_TO_CLASSIFY}"
Sentiment:"""


def ask(client, deployment, prompt):
    """Small shared helper: send one prompt, return the model's reply as text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,     # the answer here is just one word, so we cap it short
        temperature=0,     # temperature=0 -> the most consistent, least random answer
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    print("=== Zero-shot ===")
    print(ZERO_SHOT)
    print("->", ask(client, deployment, ZERO_SHOT))

    print("\n=== Few-shot (with two worked examples first) ===")
    print(FEW_SHOT)
    print("->", ask(client, deployment, FEW_SHOT))
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
