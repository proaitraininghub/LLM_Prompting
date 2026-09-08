"""
Teaching Guide Module 14 -- Output Formatting
Live-demo script.

Run:
    python part_b_prompt_engineering/module14_output_formatting.py
"""
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

UNFORMATTED_PROMPT = "List 3 planets and one fact about each."

FORMATTED_PROMPT = """List 3 planets and one fact about each.
Respond ONLY with valid JSON in exactly this shape, no extra text:
{"planets": [{"name": "...", "fact": "..."}]}"""


def main():
    client = get_client()
    deployment = get_chat_deployment()

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
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    print(raw)

    print("\n=== Parsing it as real JSON (this is why formatting matters for code) ===")
    try:
        data = json.loads(raw)
        for planet in data["planets"]:
            print(f"  {planet['name']}: {planet['fact']}")
    except json.JSONDecodeError as e:
        print(f"Could not parse as JSON: {e}")
        print("(A good live-demo moment: discuss Azure OpenAI's JSON mode /")
        print(" response_format={'type': 'json_object'} parameter for more reliability.)")


if __name__ == "__main__":
    main()
