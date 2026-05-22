# Preface

When ChatGPT first launched, some people thought LLMs were just chatbots. Chatting is only the surface—what's happening underneath is what truly matters.

An LLM is not the endpoint of conversation; it is the starting point of agency.

There is a complete chain from the first sentence you type into a dialog box to a team of agents coordinating to accomplish the task you assigned them. Every link in this chain is being studied, yet few resources explain the entire chain end to end. Many articles cover Transformers, and plenty cover Agent Loops, but very few connect "how you converse with a model," "what happens inside the model," and "how a model becomes an agent" into one coherent narrative.

This textbook does exactly that.

## What This Book Covers

The roadmap goes like this: first you learn to converse with LLMs (prompt engineering), then you understand what happens behind the conversation (APIs and inference mechanisms), next you look inside the model to see how it computes (data, architecture, attention, caching), then you step back outside to see how to make LLMs take action (agents, tools, protocols), and finally you learn to make all of this run reliably (context engineering, governance engineering).

Four parts, twenty chapters, each covering exactly one concept. Conclusions from earlier chapters are referenced and deepened in later ones—you will never encounter a concept that was never introduced.

Part One does not cover model internals. You don't need to know how an engine works to drive a car—first learn to drive well.

Part Two opens the hood. How tokens are cut, how data is cleaned, how attention is computed, how caching saves memory—these are the fundamentals.

Part Three sends the car out to do work on its own. How the Agent Loop runs, how tools are invoked, how protocols are defined, how memory is managed, how agents divide labor—these are what transform an LLM from a "chatbot" into a "capable worker."

Part Four installs the steering wheel, brakes, and dashboard. How guardrails are set, how evaluation is done, how context is managed—these ensure the system doesn't crash in production.

## How to Use This Book

Each chapter begins with a one-sentence summary and ends with references. All code examples are numbered, and you can find the corresponding runnable files in the `codes/` directory.

Chapters have prerequisites, but you don't have to read them strictly in order. If you already know how to write prompts, you can skip Chapter 1. If you already understand attention, you can skim Chapters 6 and 7. But if you're starting from scratch, read from beginning to end—later chapters build on conclusions from earlier ones.

The code is written in Python with minimal dependencies: PyTorch, tiktoken, transformers, and occasionally datasketch. Most code runs on your own machine without a GPU. A few examples require calling an LLM API (OpenAI or a local Ollama instance)—these are marked with ⚠️, and skipping them won't affect your understanding.

## What Makes This Book Different

Three things.

First, each chapter covers exactly one topic. No greed, no digressions. When covering attention, we cover attention—we won't suddenly detour into training data. When covering KV Cache, we cover KV Cache—we won't drag in positional encoding. One concept, thoroughly understood, then move on.

Second, every claim is sourced. Which paper the data came from, which experiment the numbers came from, which researcher the conclusion came from—all clearly cited. Not "some say" or "studies show," but [Vaswani et al., 2017], [Hoffmann et al., 2022]. Each chapter ends with a full reference list with arXiv links so you can download and read the originals.

Third, all code actually runs. Not pseudocode, not just function signatures without calls—complete programs you can run with `python xxx.py` and get results. Output labeled "actual output" has been verified. If your results differ, it may be due to version differences; check the comments first.

## References

The complete reference list (with key contributions and significance notes) is in [references.md](references.md).

Each chapter also has its own references, and PDF download links for core papers are in `references/papers/README.md`.

## Acknowledgments

Writing this book ultimately comes down to wanting to save others the time I once spent fumbling in the dark. People often fear what they don't understand, but a little foundational knowledge reveals that things aren't as complex as they seem.