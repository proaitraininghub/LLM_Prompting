"""
Module 14 -- Output Formatting
Live-demo script.

Run:
    python part_b_prompt_engineering/module14_output_formatting.py

What this shows: asking for free-form text vs. asking for a strict JSON
shape -- and why the JSON version matters once you want CODE (not just a
human) to read the model's answer. Run against Azure OpenAI (Option A) and
the plain OpenAI API (Option B).
"""
import sys
import os
import json
# json lets us turn a JSON-formatted text string into a real Python object
# (a dictionary/list we can loop over) -- see json.loads() below.

# Lets Python find the "common" folder at the repo root -- see
# part_a_llm_fundamentals/module02_tokenization.py for the full explanation
# of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)

# No format instructions at all -- the model is free to answer however it likes.
UNFORMATTED_PROMPT = "List 3 planets and one fact about each."

# Same request, but we explicitly demand a specific JSON shape back.
FORMATTED_PROMPT = """List 3 planets and one fact about each.
Respond ONLY with valid JSON in exactly this shape, no extra text:
{"planets": [{"name": "...", "fact": "..."}]}"""


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    print("=== No format specified ===")
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": UNFORMATTED_PROMPT}],
        max_tokens=150,
    )
    print(response.choices[0].message.content.strip())

    print("\n=== Explicit JSON format requested ===")
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": FORMATTED_PROMPT}],
        max_tokens=150,
        temperature=0,  # low temperature helps the model stick to the exact format asked for
    )
    raw = response.choices[0].message.content.strip()
    print(raw)

    print("\n=== Parsing it as real JSON (this is why formatting matters for code) ===")
    try:
        # json.loads() turns the model's text reply into an actual Python
        # dictionary -- this is the step that only works if the model's
        # output really was valid JSON, which is the whole point of this demo.
        data = json.loads(raw)
        for planet in data["planets"]:
            print(f"  {planet['name']}: {planet['fact']}")
    except json.JSONDecodeError as e:
        print(f"Could not parse as JSON: {e}")
        print("(A good live-demo moment: discuss the response_format={'type': 'json_object'}")
        print(" parameter -- supported by both Azure OpenAI and the OpenAI API -- for more reliability.)")
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
