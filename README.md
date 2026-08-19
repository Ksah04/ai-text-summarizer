# ✨ AI Text Summarizer

An AI-powered text summarization application that transforms long text into concise, readable summaries using a pretrained Transformer model.

The application uses **DistilBART**, a lightweight version of BART fine-tuned for news summarization, through the Hugging Face Transformers library. A Streamlit interface provides a simple and interactive way for users to enter text and generate summaries.

## 🌐 Live Demo

Try the deployed application here:

[AI Text Summarizer](https://ai-text-summarizer-gdrbvzeygwfhh3ft8an2ok.streamlit.app/)

---

## 📌 Project Overview

Reading and understanding large amounts of text can be time-consuming. This project demonstrates how Natural Language Processing (NLP) and Transformer-based models can be used to automatically extract the most important information from a piece of text.

The application accepts user-provided text, processes it using a pretrained Transformer model, and returns a concise summary.

For longer inputs, the application automatically divides the text into smaller token-based chunks, summarizes each chunk, combines the intermediate summaries, and performs a final summarization pass.

---

## 🚀 Features

- 🤖 AI-powered text summarization
- 🧠 Pretrained Transformer model using DistilBART
- ✂️ Automatic token-based chunking for long documents
- 🔄 Multi-stage summarization for longer inputs
- 📊 Displays input and summary statistics
- 📉 Calculates approximate percentage reduction in text
- 🎨 Interactive Streamlit interface
- ⚡ Runs locally using CPU
- ☁️ Deployed as a web application

---

## 🧠 How It Works

The application follows this pipeline:

```text
User enters text
       ↓
Tokenization
       ↓
Check input length
       ↓
 ┌───────────────┐
 │ Short text?   │
 └───────┬───────┘
         │
    ┌────┴────┐
    ↓         ↓
   YES        NO
    ↓         ↓
Summarize   Split into chunks
              ↓
       Summarize each chunk
              ↓
       Combine summaries
              ↓
       Final summarization
              ↓
         Final summary