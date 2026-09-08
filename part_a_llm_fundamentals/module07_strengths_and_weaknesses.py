"""
Teaching Guide Module 7 -- What LLMs Are Good At and Bad At
Live-demo script.

Run:
    python part_a_llm_fundamentals/module07_strengths_and_weaknesses.py

What this shows:
  1. A task LLMs are typically strong at (language/tone rewriting).
  2. A task LLMs are typically weak at (precise multi-digit arithmetic).
  3. A hallucination: confidently wrong output about something invented.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment


def ask(client, deployment, prompt, **kwargs):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    print("=== Strength: rewriting text in a different tone ===")
    print(ask(client, deployment,
              "Rewrite this in a very formal tone: 'hey can u send me that file when u get a sec'"))

    print("\n=== Weakness: precise large-number arithmetic ===")
    question = "What is 82,467 multiplied by 193,281? Answer with just the number."
    print(f"Prompt: {question}")
    answer = ask(client, deployment, question, max_tokens=30)
    real_answer = 82467 * 193281
    print(f"Model answer: {answer}")
    print(f"Real answer:  {real_answer:,}")
    print("(Run this a few times -- the model's confidence rarely matches its accuracy here.)")

    print("\n=== Hallucination: asking about something that doesn't exist ===")
    print(ask(client, deployment,
              "Summarize the plot of the 2014 novel 'The Glass Orchard' by Miriam Delacroix.",
              max_tokens=150))
    print("(There is no such novel or author -- watch how confidently the model may still answer.)")


if __name__ == "__main__":
    main()
