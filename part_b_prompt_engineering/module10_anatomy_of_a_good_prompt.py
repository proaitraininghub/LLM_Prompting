"""
Teaching Guide Module 10 -- The Anatomy of a Good Prompt
(Role, Task, Context, Format, Constraints)
Live-demo script.

Run:
    python part_b_prompt_engineering/module10_anatomy_of_a_good_prompt.py

What this shows: the same request, once as a vague one-liner and once
built out with Role/Task/Context/Format/Constraints -- side by side.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment

WEAK_PROMPT = "Help me with my math homework."

STRONG_PROMPT = """Role: You are a patient high-school math tutor.
Task: Help the student solve the equation 3x + 7 = 22, explaining each step.
Context: The student is a beginner and gets confused by algebra notation.
Format: Number each step, and end with the final answer on its own line.
Constraints: Do not skip any step, and keep the language simple."""


def ask(client, deployment, prompt):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def main():
    client = get_client()
    deployment = get_chat_deployment()

    print("=== Weak prompt ===")
    print(f'"{WEAK_PROMPT}"\n')
    print(ask(client, deployment, WEAK_PROMPT))

    print("\n\n=== Strong prompt (Role / Task / Context / Format / Constraints) ===")
    print(STRONG_PROMPT, "\n")
    print(ask(client, deployment, STRONG_PROMPT))


if __name__ == "__main__":
    main()
