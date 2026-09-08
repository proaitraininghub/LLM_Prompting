"""
Teaching Guide Module 13 -- Role and Persona Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module13_role_and_persona.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

QUESTION = "Why is the sky blue?"

PERSONAS = [
    "You are a physics professor speaking to graduate students.",
    "You are explaining this to a curious 6-year-old.",
    "You are a pirate captain who explains everything in pirate slang.",
]


def main():
    client = get_client()
    deployment = get_chat_deployment()

    for persona in PERSONAS:
        print(f"=== Persona: {persona} ===")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": QUESTION},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        print(response.choices[0].message.content.strip(), "\n")


if __name__ == "__main__":
    main()
