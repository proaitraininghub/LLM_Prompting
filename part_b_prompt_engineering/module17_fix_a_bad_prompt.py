"""
Module 17 -- A Worked Example: Improving a Bad Prompt Step by Step
Live-demo script.

Run:
    python part_b_prompt_engineering/module17_fix_a_bad_prompt.py

Starts from "Tell me about marketing." and improves it in stages -- adding
a role, then a specific task, then format and constraints -- running each
version live so students see the output change at every stage. This
deliberately reuses ideas from earlier modules (role/persona from Module 13,
the Role/Task/Context/Format/Constraints structure from Module 10) to show
them working together in one worked example. Run against Azure OpenAI
(Option A) and the plain OpenAI API (Option B).
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

# Each stage adds ONE more improvement on top of the previous stage's
# prompt -- read them in order to see the prompt grow.
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
    """Small shared helper: send one prompt, return the model's reply as text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.5,  # moderate temperature -- some natural variety, still fairly consistent
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")
    for title, prompt in STAGES:
        print(f"=== {title} ===")
        print(f'Prompt: "{prompt}"\n')
        print(ask(client, deployment, prompt))
        print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
