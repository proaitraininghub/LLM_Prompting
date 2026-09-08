"""
Teaching Guide Module 4 -- Next-Token Prediction and Temperature
Live-demo script.

Run:
    python part_a_llm_fundamentals/module04_next_token_and_temperature.py

What this shows:
  1. The model predicting one token at a time (via logprobs), showing the
     actual probability distribution it samples from.
  2. How temperature changes how "risky" that sampling is -- same prompt,
     multiple temperatures, run twice each so you can see the variation live.

Note: logprobs support requires a chat-completions-capable deployment
(e.g. gpt-4o-mini or newer). If your deployment doesn't support logprobs,
the first demo will print a friendly error and the temperature demo will
still run fine.
"""
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import get_client, get_chat_deployment


def show_next_token_probabilities(client, deployment):
    prompt = "The sky is"
    print(f'=== Next-token probabilities after: "{prompt}" ===\n')
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": f"Complete this sentence with exactly one word: '{prompt}'"}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            temperature=0,
        )
        top_choices = response.choices[0].logprobs.content[0].top_logprobs
        for choice in top_choices:
            probability_pct = round(math.exp(choice.logprob) * 100, 2)
            print(f"  token: {choice.token!r:12s} probability: {probability_pct}%")
    except Exception as e:
        print(f"(Could not fetch logprobs on this deployment: {e})")
        print("Skipping to the temperature demo below.")


def show_temperature_effect(client, deployment):
    prompt = "Write a one-sentence opening line for a mystery novel."
    print("\n=== Same prompt, different temperatures (run twice each) ===\n")
    for temp in [0.0, 0.7, 1.2]:
        print(f"--- temperature = {temp} ---")
        for run in range(2):
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=40,
            )
            print(f"  Run {run + 1}: {response.choices[0].message.content.strip()}")
        print()
    print("Notice temperature=0.0 gives (close to) the same answer both times --")
    print("higher temperatures introduce more variety, at the cost of predictability.")


if __name__ == "__main__":
    client = get_client()
    deployment = get_chat_deployment()
    show_next_token_probabilities(client, deployment)
    show_temperature_effect(client, deployment)
