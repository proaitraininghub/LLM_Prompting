"""
Teaching Guide Module 13 -- Role and Persona Prompting
Live-demo script.

Run:
    python part_b_prompt_engineering/module13_role_and_persona.py

What this shows: the exact same question, answered three times with three
different "personas" assigned to the model via the system message. Run
against Azure OpenAI (Option A) and the plain OpenAI API (Option B).
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

QUESTION = "Why is the sky blue?"

# Each persona is set via the "system" message -- a special instruction
# that shapes HOW the model behaves for the rest of the conversation,
# separate from the actual user question.
PERSONAS = [
    "You are a physics professor speaking to graduate students.",
    "You are explaining this to a curious 6-year-old.",
    "You are a pirate captain who explains everything in pirate slang.",
]


def main(label, client, deployment):
    print(f"#### {label} ####\n")
    for persona in PERSONAS:
        print(f"=== Persona: {persona} ===")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": persona},   # sets the "who am I" role
                {"role": "user", "content": QUESTION},     # the actual question, unchanged each time
            ],
            max_tokens=100,
            temperature=0.7,  # a little randomness so each persona's voice can show through
        )
        print(response.choices[0].message.content.strip(), "\n")


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
