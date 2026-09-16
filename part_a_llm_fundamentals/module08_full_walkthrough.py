"""
Teaching Guide Module 8 -- A Worked Walkthrough: Tracing One Prompt Start to Finish
Live-demo script.

Run:
    python part_a_llm_fundamentals/module08_full_walkthrough.py

Traces a single prompt through every stage covered in Part A: tokenization,
the model generating a response, temperature's effect on that generation,
and the context window that would hold the exchange -- ending with the
token usage the API actually reports. Run against Azure OpenAI (Option A)
and the plain OpenAI API (Option B).
"""
import sys
import os

# Lets Python find the "common" folder at the repo root -- see
# module02_tokenization.py for the full explanation of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)

PROMPT = "Explain gravity simply."


def main(label, client, deployment):
    print(f"#### {label} ####\n")
    enc = tiktoken.get_encoding("cl100k_base")

    print("=== Step 1: Tokenize the prompt ===")
    print(f'Prompt: "{PROMPT}"')
    tokens = enc.encode(PROMPT)  # this happens locally -- no API call yet
    print(f"Tokens ({len(tokens)}): {[enc.decode([t]) for t in tokens]}\n")

    print("=== Step 2: Send to the model, low temperature (more predictable) ===")
    response_low = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.0,   # low temperature -> more consistent, "safe" answers
        max_tokens=80,
    )
    print(response_low.choices[0].message.content.strip())
    # Every response includes a "usage" object reporting exactly how many
    # tokens the prompt used, how many the reply used, and the total --
    # this is what you'd actually be billed for.
    u = response_low.usage
    print(f"\nToken usage -- prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}\n")

    print("=== Step 3: Same prompt, high temperature (more varied) ===")
    response_high = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=1.2,   # high temperature -> more creative, less predictable
        max_tokens=80,
    )
    print(response_high.choices[0].message.content.strip())

    print("\n=== Step 4: The context window would hold this whole exchange ===")
    print("If a student asked a follow-up like 'Explain it again for a 5-year-old',")
    print("the model would see this entire exchange again as part of its context --")
    print("nothing here is remembered outside of what's resent in the messages list.")
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
