"""
Teaching Guide Module 15 -- Iterative Refinement
Live-demo script.

Run:
    python part_b_prompt_engineering/module15_iterative_refinement.py

What this shows: treating a first response as a draft, then asking the
model to revise it -- twice -- watching the output change each time,
with the full conversation history carried forward each turn. Run against
Azure OpenAI (Option A) and the plain OpenAI API (Option B).
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


def ask(client, deployment, messages, max_tokens=200):
    """
    Small shared helper: send the FULL conversation-so-far (not just the
    latest message) and return the model's reply as text. Sending the whole
    "messages" list each time is what lets the model "remember" the earlier
    drafts when we ask it to revise -- see main() below.
    """
    response = client.chat.completions.create(
        model=deployment, messages=messages, max_tokens=max_tokens, temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    # We build "messages" up turn by turn, appending both our own requests
    # AND the model's previous replies -- that's what makes each new request
    # a genuine "refinement" of what came before, not a fresh, unrelated ask.
    messages = [{"role": "user", "content": "Write a short product description for a smart water bottle."}]
    print("=== Draft 1 ===")
    draft1 = ask(client, deployment, messages)
    print(draft1)
    messages.append({"role": "assistant", "content": draft1})  # remember the model's own reply

    messages.append({"role": "user", "content": "Make it punchier and add one call to action."})
    print("\n=== Draft 2 (refinement 1) ===")
    draft2 = ask(client, deployment, messages)
    print(draft2)
    messages.append({"role": "assistant", "content": draft2})

    messages.append({"role": "user", "content": "Now shorten it to under 30 words."})
    print("\n=== Draft 3 (refinement 2) ===")
    draft3 = ask(client, deployment, messages)
    print(draft3)
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
