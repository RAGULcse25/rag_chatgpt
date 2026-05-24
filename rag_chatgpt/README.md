---
title: RAG ChatGPT Assistant
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG-powered document Q&A assistant with local LLM fallback
---

# 🧠 RAG Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from your documents.

## Features
- 💬 Multi-turn chat with conversation history
- 📚 Document browser with chunk search
- 🔒 Local LLM fallback (distilgpt2) when OpenAI quota is exhausted
- ⚙️ Adjustable Top-K, temperature, and max tokens
- 📊 Index stats and relevance scoring
- 💾 Export chat history as JSON

## Setup
Set `OPENAI_API_KEY` in Space Secrets for OpenAI-powered answers.
Without it, the app automatically uses the local model.
