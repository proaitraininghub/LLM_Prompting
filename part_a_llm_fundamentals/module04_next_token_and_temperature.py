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
  Both shown against Azure OpenAI (Option A) and the plain OpenAI API (Option B).

Note: logprobs support requires a chat-completions-capable deployment
(e.g. gpt-4o-mini or newer). If your deployment doesn't support logprobs,
the first demo will print a friendly error and the temperature demo will
still run fine.
"""
import sys
import os
import math
# math.exp() below converts a "log probability" back into an ordinary
# percentage -- models report probabilities in log form for technical
# reasons, but percentages are much easier for humans to read.

# Lets Python find the "common" folder at the repo root -- see
# module02_tokenization.py for the full explanation of this line.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.client import (
    get_azure_client, get_azure_chat_deployment,
    get_openai_client, get_openai_chat_model,
)


def show_next_token_probabilities(client, deployment):
    """
    Ask the model to predict just ONE token, but request "logprobs" --
    which means "also tell me the other tokens you considered, and how
    likely each one was." This lets us peek at the model's actual
    decision-making instead of only seeing its final choice.
    """
    prompt = "The sky is"
    print(f'=== Next-token probabilities after: "{prompt}" ===\n')
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": f"Complete this sentence with exactly one word: '{prompt}'"}],
            max_tokens=1,       # we only want to see the very next token, not a full sentence
            logprobs=True,      # "yes, include probability information in the response"
            top_logprobs=5,     # "show me the top 5 tokens it considered, not just the winner"
            temperature=0,      # temperature=0 means "always pick the single most likely token"
                                 # (see show_temperature_effect below for what changing this does)
        )
        # Dig into the response structure to get that list of top candidate tokens.
        top_choices = response.choices[0].logprobs.content[0].top_logprobs
        for choice in top_choices:
            # logprob is a "log probability" -- math.exp() converts it back
            # into an ordinary 0-100% probability that's easier to read.
            probability_pct = round(math.exp(choice.logprob) * 100, 2)
            print(f"  token: {choice.token!r:12s} probability: {probability_pct}%")
    except Exception as e:
        # Not every deployed model supports logprobs -- if this one doesn't,
        # fail gracefully instead of crashing the whole script.
        print(f"(Could not fetch logprobs on this deployment: {e})")
        print("Skipping to the temperature demo below.")


def show_temperature_effect(client, deployment):
    """
    "Temperature" controls how much randomness the model uses when picking
    its next word. Low temperature (near 0) = very predictable, almost
    always the same answer. High temperature (above 1) = more creative and
    varied, but also less consistent. We send the SAME prompt at three
    different temperatures, twice each, so you can watch the variation
    (or lack of it) happen live.
    """
    prompt = "Write a one-sentence opening line for a mystery novel."
    print("\n=== Same prompt, different temperatures (run twice each) ===\n")
    for temp in [0.0, 0.7, 1.2]:
        print(f"--- temperature = {temp} ---")
        for run in range(2):  # run each temperature twice, back to back
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
    print("#### Option A: Azure OpenAI ####\n")
    azure_client = get_azure_client()
    azure_deployment = get_azure_chat_deployment()
    show_next_token_probabilities(azure_client, azure_deployment)
    show_temperature_effect(azure_client, azure_deployment)

    print("\n#### Option B: OpenAI API ####\n")
    openai_client = get_openai_client()
    openai_model = get_openai_chat_model()
    show_next_token_probabilities(openai_client, openai_model)
    show_temperature_effect(openai_client, openai_model)
