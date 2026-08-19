# AI Text Summarizer

A simple AI-powered text summarization app built with Python, Hugging Face Transformers, and Streamlit.

I built this project to get more practical experience working with pretrained AI models and to understand what happens between giving a model some text and getting a useful result back.

## Live Demo

[Try the AI Text Summarizer](https://ai-text-summarizer-gdrbvzeygwfhh3ft8an2ok.streamlit.app/)

## What it does

The app takes a piece of text and generates a shorter version that keeps the main points.

It works with both shorter and longer pieces of text. For longer inputs, the text is split into smaller chunks before being summarized.

## How it works

The project uses the `sshleifer/distilbart-cnn-12-6` model from Hugging Face.

The basic flow is:

```text
Text entered by user
        ↓
Tokenization
        ↓
Check text length
        ↓
Short text → Summarize directly
        ↓
Long text → Split into chunks
        ↓
Summarize each chunk
        ↓
Combine the summaries
        ↓
Generate final summary