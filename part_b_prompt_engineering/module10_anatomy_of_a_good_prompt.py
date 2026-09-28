"""
Module 10 -- The Anatomy of a Good Prompt
(Role, Task, Context, Format, Constraints)


Run:
    python part_b_prompt_engineering/module10_anatomy_of_a_good_prompt.py

What this shows: the same request, once as a vague one-liner and once
built out with Role/Task/Context/Format/Constraints -- side by side.
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

# A deliberately vague, one-line prompt -- the kind a beginner might write.
WEAK_PROMPT = "tell students about india"

# The same underlying request, rebuilt with five ingredients of a good prompt:
#   Role        -- who should the model "act as"?
#   Task        -- exactly what should it do?
#   Context     -- what background does it need to know?
#   Format      -- how should the answer be structured?
#   Constraints -- what rules must it follow?
STRONG_PROMPT = "role : you are a history professor teaching a class of college students about india. " \
"task : provide a brief overview of the history of india, " \
"highlighting key events and figures. context : the students have a basic understanding of world history but are not familiar with indian history. " \
"format : present the information in a clear and organized manner, using bullet points for each key event or figure. c" \
"onstraints : keep the overview concise, no more than 200 words, and avoid using overly complex language." 

def ask(client, deployment, prompt):
    """Small shared helper: send one prompt, return the model's reply as text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    print("=== Weak prompt ===")
    print(f'"{WEAK_PROMPT}"\n')
    print(ask(client, deployment, WEAK_PROMPT))

    print("\n\n=== Strong prompt (Role / Task / Context / Format / Constraints) ===")
    print(STRONG_PROMPT, "\n")
    print(ask(client, deployment, STRONG_PROMPT))
    print()


if __name__ == "__main__":
    #main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
