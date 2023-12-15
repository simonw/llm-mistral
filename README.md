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
You can now access the three Mistral hosted models: `mistral-tiny`, `mistral-small` and `mistral-medium`.

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

## Options

All three models accept the following options, using `-o name value` syntax:

- `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher increases randomness, lower values are more focused and deterministic.
- `-o top_p 0.1`: 0.1 means consider only tokens in the top 10% probability mass. Use this or temperature but not both.
- `-o max_tokens 20`: Maximum number of tokens to generate in the completion.
- `-o safe_mode 1`: Turns on [safe mode](https://docs.mistral.ai/platform/guardrailing/), which adds a system prompt to add guardrails to the model output.
- `-o random_seed 123`: Set an integer random seed to generate deterministic results.

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
