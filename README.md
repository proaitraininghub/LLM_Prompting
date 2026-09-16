# LLM Fundamentals + Prompt Engineering — Live-Demo Scripts

One small, standalone Python script per teaching stage, so you can run it
live in class and let students see actual model output at each concept.

**Every script demonstrates its concept two ways, back to back:** first
against Azure OpenAI / Microsoft Foundry (Option A), then against the plain
OpenAI API (Option B). Both code paths are always in the file, side by side
— nothing to toggle. This is deliberate: students see the exact same
`client.chat.completions.create(...)` call work against two different
backends, which is a good moment to point out that most LLM application code
isn't tied to one vendor.

You only need to fill in the credentials for whichever provider(s) you plan
to demo — see `.env.example`. If you only fill in the Azure section, Option
A will run fine and Option B will raise a clear "missing environment
variable" error when it gets to that part; just comment out that line in the
script's `__main__` block if you don't want to demo it.

Every script's filename and docstring reference the matching module number
in **LLM_Fundamentals_Teaching_Guide.docx**, so you can pull up the script
right when you reach that module.

## Setup (do this once)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in:
   - Your Foundry project's endpoint, API key, and deployment names
     (Azure AI course, Module 4: Deploying Your First Model) for Option A, and/or
   - Your OpenAI API key (from platform.openai.com/api-keys) and model names
     for Option B.
3. Confirm you have a chat model deployed/available (e.g. `gpt-4o-mini`) and,
   if you want to run `module03_embeddings.py`, an embedding model too
   (e.g. `text-embedding-3-small`).

## Running a script

From the repo root:
```
python part_a_llm_fundamentals/module02_tokenization.py
python part_b_prompt_engineering/module10_anatomy_of_a_good_prompt.py
```

Each script is self-contained — run it directly, read the printed output
live with the class, and re-run as many times as you like (several scripts
are designed to be re-run to show variation, e.g. the temperature demo).
Output is grouped under "Option A: Azure OpenAI" and "Option B: OpenAI API"
headers so it's easy to point out which backend produced which block.

## Script map

### Part A — LLM Fundamentals

| Script | Teaching Guide Module | What it demonstrates |
|---|---|---|
| `module02_tokenization.py` | Module 2 | Splitting text into tokens; confirming the count against a real API call |
| `module03_embeddings.py` | Module 3 | Turning words into vectors; cosine similarity between related/unrelated pairs |
| `module04_next_token_and_temperature.py` | Module 4 | Live next-token probabilities (logprobs); temperature's effect on generation |
| `module05_context_window.py` | Module 5 | Token count growing turn by turn; recall within vs. outside the window |
| `module07_strengths_and_weaknesses.py` | Module 7 | A strength (tone rewriting), a weakness (large arithmetic), a hallucination |
| `module08_full_walkthrough.py` | Module 8 | One prompt traced through tokenization, generation, temperature, and usage |

*(Module 1 "What Is an LLM" and Module 6 "How LLMs Are Trained" are
conceptual/whiteboard modules — there's no corresponding live-demo script,
since pretraining/fine-tuning/RLHF aren't things you can demo against a
deployed model.)*

### Part B — Prompt Engineering

| Script | Teaching Guide Module | What it demonstrates |
|---|---|---|
| `module10_anatomy_of_a_good_prompt.py` | Module 10 | Vague prompt vs. Role/Task/Context/Format/Constraints, side by side |
| `module11_zero_shot_vs_few_shot.py` | Module 11 | Same classification task, zero-shot vs. with two worked examples |
| `module12_chain_of_thought.py` | Module 12 | Direct answer vs. "think step by step" on a multi-step arithmetic problem |
| `module13_role_and_persona.py` | Module 13 | Same question, three different assigned personas |
| `module14_output_formatting.py` | Module 14 | Free-form output vs. an explicit JSON format you can actually parse |
| `module15_iterative_refinement.py` | Module 15 | Three drafts of the same text, each refined from the last |
| `module17_fix_a_bad_prompt.py` | Module 17 | "Tell me about marketing." improved across four stages |

## Putting this in your own GitHub repo

This folder is already a git repository with one initial commit. To push it
to your own GitHub account (same steps as the GitHub Basics course):

```
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Then on any other machine:
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
cp .env.example .env   # then fill in your own values
```

**Reminder:** `.env` is in `.gitignore` on purpose — never commit your real
API key. If you ever paste a key into a script or commit by accident, rotate
it in the Foundry portal immediately.
