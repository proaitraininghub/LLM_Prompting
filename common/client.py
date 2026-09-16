"""
common/client.py
=================

WHAT THIS FILE IS FOR (read this first if Python is new to you):

Every demo script in this repo needs to "log in" to an AI model before it
can ask it anything. That login process is a few lines of setup code --
and it's the SAME few lines every single script would otherwise need to
repeat. Instead, we write that setup code ONCE, here, and every other
script just borrows it. This is a very common pattern in real software:
put shared, reusable logic in one place instead of copy-pasting it everywhere.

This file supports TWO different ways of reaching an AI model:

  Option A: Azure OpenAI / Microsoft Foundry  (functions starting get_azure_...)
  Option B: the plain OpenAI API              (functions starting get_openai_...)

They do the same job -- get you a "client" (an object you can use to send
requests to the model) -- just talking to two different companies' servers.
Every demo script in this repo calls BOTH, one after the other, so you can
see the exact same kind of request work against either one.
"""

# --- Imports: bringing in code that already exists, instead of writing it ourselves ---

import os
# "os" lets Python read information from your computer's operating system --
# here, we use it to read environment variables (see below).

from openai import AzureOpenAI, OpenAI
# This brings in two ready-made "client" classes from the openai package
# (which we installed via requirements.txt). AzureOpenAI knows how to talk
# to Azure OpenAI / Foundry; OpenAI knows how to talk to the plain OpenAI API.
# We don't have to write any networking code ourselves -- this package
# already knows how to send a request and get a response back.

from dotenv import load_dotenv
# This lets us read values out of a ".env" file (a small text file where
# you keep secrets like API keys, so they never get typed directly into
# your code or committed to GitHub).

load_dotenv()
# Actually reads your .env file now, and makes everything in it available
# through os.environ (see _check_env and the get_* functions below).


# --- What each "way" of connecting needs to have been filled in ---

# If you're using Azure OpenAI, your .env file must have these three values:
AZURE_REQUIRED_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT"]

# If you're using the plain OpenAI API, your .env file must have these two:
OPENAI_REQUIRED_VARS = ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL"]


def _check_env(required_vars, hint):
    """
    Before we try to connect, double check the .env file actually has
    everything we need. If something's missing, stop here with a clear,
    friendly error message instead of letting the script fail later with
    a confusing network error.

    (The underscore at the start of the function name, _check_env, is a
    Python convention meaning "this is a helper function for use inside
    this file only -- other scripts shouldn't need to call it directly.")
    """
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing) + "\n" + hint
        )


# ============================================================================
# Option A: Azure OpenAI / Microsoft Foundry
# ============================================================================

def get_azure_client() -> AzureOpenAI:
    """
    Builds and returns an AzureOpenAI client -- the object every script
    uses to actually send a request to the model. Think of this function
    as "log in to Azure OpenAI and hand me back a ready-to-use connection."
    """
    _check_env(
        AZURE_REQUIRED_VARS,
        "Copy .env.example to .env and fill in your Azure OpenAI / Foundry project details.",
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],   # the web address of your Azure resource
        api_key=os.environ["AZURE_OPENAI_API_KEY"],           # your secret key, proves it's really you
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2026-01-01-preview"),  # which version of Azure's API to speak
    )


def get_azure_chat_deployment() -> str:
    """
    Returns the DEPLOYMENT NAME you chose in the Azure/Foundry portal when
    you deployed a chat model (this is a name you picked, like "my-gpt4o" --
    not the underlying model name itself). Every chat request needs to say
    which deployment it wants to talk to, so scripts call this to get that name.
    """
    _check_env(AZURE_REQUIRED_VARS, "Set AZURE_OPENAI_CHAT_DEPLOYMENT in .env.")
    return os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]


def get_azure_embedding_deployment() -> str:
    """
    Same idea as get_azure_chat_deployment(), but for the embedding model
    (used only by module03_embeddings.py, which turns words into number
    vectors). Falls back to a sensible default name if you haven't set one.
    """
    return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


# ============================================================================
# Option B: plain OpenAI API
# ============================================================================

def get_openai_client() -> OpenAI:
    """
    Same job as get_azure_client(), but for the plain OpenAI API instead
    of Azure. Notice this one only needs an API key -- no endpoint URL --
    because the plain OpenAI API always lives at the same fixed address,
    unlike Azure where every customer gets their own resource URL.
    """
    _check_env(
        OPENAI_REQUIRED_VARS,
        "Copy .env.example to .env and fill in OPENAI_API_KEY (from platform.openai.com/api-keys) "
        "and OPENAI_CHAT_MODEL.",
    )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openai_chat_model() -> str:
    """
    Returns the MODEL NAME to use with the plain OpenAI API (e.g. "gpt-4o-mini").
    Note this is different from Azure's "deployment name" idea above --
    with plain OpenAI you just say which model you want directly, you don't
    deploy it yourself first.
    """
    _check_env(OPENAI_REQUIRED_VARS, "Set OPENAI_CHAT_MODEL in .env, e.g. gpt-4o-mini.")
    return os.environ["OPENAI_CHAT_MODEL"]


def get_openai_embedding_model() -> str:
    """Same idea as get_openai_chat_model(), but for embeddings."""
    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================================
# Older names kept working, in case any script still uses them
# ============================================================================
# These three lines just say "if some script asks for get_client(), quietly
# give it get_azure_client() instead" -- so nothing breaks even if a script
# was written before this file supported two providers.
get_client = get_azure_client
get_chat_deployment = get_azure_chat_deployment
get_embedding_deployment = get_azure_embedding_deployment
