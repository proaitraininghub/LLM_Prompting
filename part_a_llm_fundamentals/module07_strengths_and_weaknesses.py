"""
Teaching Guide Module 7 -- What LLMs Are Good At and Bad At
Live-demo script.

Run:
    python part_a_llm_fundamentals/module07_strengths_and_weaknesses.py

What this shows:
  1. A task LLMs are typically strong at (language/tone rewriting).
  2. A task LLMs are typically weak at (precise multi-digit arithmetic).
  3. A hallucination: confidently wrong output about something invented.
  Run against Azure OpenAI (Option A) and the plain OpenAI API (Option B).
"""
import sys
import os

# Lets Python find the "common" folder at the repo root -- see
# module02_tokenization.py for the full explanation of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)


def ask(client, deployment, prompt, **kwargs):
    """
    A small shared helper: send one prompt, get back the model's reply as
    plain text. We'll call this three times below with three different
    prompts, instead of repeating the same five lines of API-calling code
    three separate times.
    (**kwargs means "accept any extra named options and pass them straight
    through to the API call" -- e.g. max_tokens=30 in the arithmetic demo below.)
    """
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content.strip()


def main(label, client, deployment):
    print(f"#### {label} ####\n")

    print("=== Strength: rewriting text in a different tone ===")
    print(ask(client, deployment,
              "Rewrite this in a very formal tone: 'hey can u send me that file when u get a sec'"))

    print("\n=== Weakness: precise large-number arithmetic ===")
    question = "What is 82,467 multiplied by 193,281? Answer with just the number."
    print(f"Prompt: {question}")
    answer = ask(client, deployment, question, max_tokens=30)
    real_answer = 82467 * 193281  # Python computes this exactly -- a good comparison point
    print(f"Model answer: {answer}")
    print(f"Real answer:  {real_answer:,}")
    print("(Run this a few times -- the model's confidence rarely matches its accuracy here.)")

    print("\n=== Hallucination: asking about something that doesn't exist ===")
    print(ask(client, deployment,
              "Summarize the plot of the 2014 novel 'The Glass Orchard' by Miriam Delacroix.",
              max_tokens=150))
    print("(There is no such novel or author -- watch how confidently the model may still answer.)")
    print()


if __name__ == "__main__":
    main("Option A: Azure OpenAI", get_azure_client(), get_azure_chat_deployment())
    main("Option B: OpenAI API", get_openai_client(), get_openai_chat_model())
