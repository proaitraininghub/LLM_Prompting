"""
Teaching Guide Module 17 -- A Worked Example: Improving a Bad Prompt Step by Step
Live-demo script.

Run:
    python part_b_prompt_engineering/module17_fix_a_bad_prompt.py

Starts from "Tell me about marketing." and improves it in stages -- adding
a role, then a specific task, then format and constraints -- running each
version live so students see the output change at every stage.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

STAGES = [
    ("Stage 0 -- the bad prompt",
     "Tell me about marketing."),

    ("Stage 1 -- add a role and audience",
     "You are a marketing consultant advising a small local bakery. "
     "Tell me about marketing."),

    ("Stage 2 -- add a specific task",
     "You are a marketing consultant advising a small local bakery. "
     "Suggest 3 low-cost marketing ideas they could start this month."),

    ("Stage 3 -- add format and constraints",
     "You are a marketing consultant advising a small local bakery. "
     "Suggest exactly 3 low-cost marketing ideas they could start this month. "
     "Format as a numbered list, one sentence each, no more than 20 words per idea."),
]


def ask(client, deployment, prompt, max_tokens=180):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    for title, prompt in STAGES:
        print(f"=== {title} ===")
        print(f'Prompt: "{prompt}"\n')
        print(ask(client, deployment, prompt))
        print()


if __name__ == "__main__":
    main()
