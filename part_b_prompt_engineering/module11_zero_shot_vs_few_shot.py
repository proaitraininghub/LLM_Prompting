"""
Teaching Guide Module 11 -- Zero-Shot vs. Few-Shot Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module11_zero_shot_vs_few_shot.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

REVIEW_TO_CLASSIFY = "The screen cracked after one week and support never replied."

ZERO_SHOT = f'Classify the sentiment of this review as positive, negative, or neutral: "{REVIEW_TO_CLASSIFY}"'

FEW_SHOT = f"""Classify each review's sentiment as positive, negative, or neutral.

Review: "Fast shipping and works perfectly!"
Sentiment: positive

Review: "It's okay, does the job."
Sentiment: neutral

Review: "{REVIEW_TO_CLASSIFY}"
Sentiment:"""


def ask(client, deployment, prompt):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    print("=== Zero-shot ===")
    print(ZERO_SHOT)
    print("->", ask(client, deployment, ZERO_SHOT))

    print("\n=== Few-shot (with two worked examples first) ===")
    print(FEW_SHOT)
    print("->", ask(client, deployment, FEW_SHOT))


if __name__ == "__main__":
    main()
