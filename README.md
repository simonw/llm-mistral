# llm-mistral

[![PyPI](https://img.shields.io/pypi/v/llm-mistral.svg)](https://pypi.org/project/llm-mistral/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-mistral?include_prereleases&label=changelog)](https://github.com/simonw/llm-mistral/releases)
[![Tests](https://github.com/simonw/llm-mistral/workflows/Test/badge.svg)](https://github.com/simonw/llm-mistral/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-mistral/blob/main/LICENSE)

LLM plugin providing access to Mistral models busing the Mistral API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

    llm install llm-mistral

## Usage

Usage instructions go here.

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
