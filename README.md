# llm-mistral

[![PyPI](https://img.shields.io/pypi/v/llm-mistral.svg)](https://pypi.org/project/llm-mistral/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-mistral?include_prereleases&label=changelog)](https://github.com/simonw/llm-mistral/releases)
[![Tests](https://github.com/simonw/llm-mistral/workflows/Test/badge.svg)](https://github.com/simonw/llm-mistral/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-mistral/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin providing access to [Mistral](https://mistral.ai) models using the Mistral API

## Installation

Install this plugin in the same environment as LLM:
```bash
llm install llm-mistral
```
## Usage

First, obtain an API key for [the Mistral API](https://console.mistral.ai/).

Configure the key using the `llm keys set mistral` command:
```bash
llm keys set mistral
```
```
<paste key here>
```
You can now access the Mistral hosted models. Run `llm models` for a list.

To run a prompt through `mistral-tiny`:

```bash
llm -m mistral-tiny 'A sassy name for a pet sasquatch'
```
To start an interactive chat session with `mistral-small`:
```bash
llm chat -m mistral-small
```
```
Chatting with mistral-small
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> three proud names for a pet walrus
1. "Nanuq," the Inuit word for walrus, which symbolizes strength and resilience.
2. "Sir Tuskalot," a playful and regal name that highlights the walrus' distinctive tusks.
3. "Glacier," a name that reflects the walrus' icy Arctic habitat and majestic presence.
```
To use a system prompt with `mistral-medium` to explain some code:
```bash
cat example.py | llm -m mistral-medium -s 'explain this code'
```
## Vision

The `pixtral-12b` model is capable of interpreting images. You can use that like this:

```bash
llm -m pixtral-12b 'describe this image' -a https://static.simonwillison.net/static/2024/earth.jpg
```
You can also pass filenames instead of URLs.

## Model options

All three models accept the following options, using `-o name value` syntax:

- `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher increases randomness, lower values are more focused and deterministic.
- `-o top_p 0.1`: 0.1 means consider only tokens in the top 10% probability mass. Use this or temperature but not both.
- `-o max_tokens 20`: Maximum number of tokens to generate in the completion.
- `-o safe_mode 1`: Turns on [safe mode](https://docs.mistral.ai/platform/guardrailing/), which adds a system prompt to add guardrails to the model output.
- `-o random_seed 123`: Set an integer random seed to generate deterministic results.

## Available models

Run `llm models` for a full list of Mistral models. This plugin configures the following alias shortcuts:

<!-- [[[cog
import cog, json
from llm_mistral import DEFAULT_ALIASES
for model_id, alias in DEFAULT_ALIASES.items():
    cog.out(f"- `{alias}` for `{model_id}`\n")
]]] -->
- `mistral-tiny` for `mistral/mistral-tiny`
- `mistral-nemo` for `mistral/open-mistral-nemo`
- `mistral-small` for `mistral/mistral-small`
- `mistral-medium` for `mistral/mistral-medium`
- `mistral-large` for `mistral/mistral-large-latest`
- `codestral-mamba` for `mistral/codestral-mamba-latest`
- `codestral` for `mistral/codestral-latest`
- `ministral-3b` for `mistral/ministral-3b-latest`
- `ministral-8b` for `mistral/ministral-8b-latest`
- `pixtral-12b` for `mistral/pixtral-12b-latest`
- `pixtral-large` for `mistral/pixtral-large-latest`
<!-- [[[end]]] -->


## Refreshing the model list

Mistral sometimes release new models.

To make those models available to an existing installation of `llm-mistral` run this command:
```bash
llm mistral refresh
```
This will fetch and cache the latest list of available models. They should then become available in the output of the `llm models` command.

## Embeddings

The Mistral [Embeddings API](https://docs.mistral.ai/platform/client#embeddings) can be used to generate 1,024 dimensional embeddings for any text.

To embed a single string:

```bash
llm embed -m mistral-embed -c 'this is text'
```
This will return a JSON array of 1,024 floating point numbers.

The [LLM documentation](https://llm.datasette.io/en/stable/embeddings/index.html) has more, including how to embed in bulk and store the results in a SQLite database.

See [LLM now provides tools for working with embeddings](https://simonwillison.net/2023/Sep/4/llm-embeddings/) and [Embeddings: What they are and why they matter](https://simonwillison.net/2023/Oct/23/embeddings/) for more about embeddings.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-mistral
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```
