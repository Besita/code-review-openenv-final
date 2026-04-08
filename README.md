🚀 Code Review OpenEnv Environment
📌 Overview

This project implements a real-world OpenEnv environment for automated code review.
An agent analyzes code, detects issues, assigns severity, suggests fixes, and provides reasoning.

The environment evaluates the agent using semantic similarity and multi-factor reward shaping, enabling realistic and granular feedback.

🎯 Motivation

Code review is a critical task in software engineering. This environment simulates realistic scenarios including:

Security vulnerabilities (e.g., SQL injection)
Runtime errors (e.g., division by zero)
Performance and code quality issues (e.g., inefficient loops)

It allows testing agent performance in a controlled, reproducible setting.

🧠 Action Space
{
  "issues": ["list of detected issues"],
  "severity": "low | medium | high",
  "suggestion": "fix recommendation",
  "reasoning": "explanation of the issue"
}
📊 Observation Space
{
  "code": "input code snippet",
  "score": "float (0.0 to 1.0)",
  "feedback": "detailed evaluation breakdown"
}
🧩 Tasks
Difficulty	Description	Example Issue
Easy	Basic runtime errors	division by zero
Medium	Security issues	SQL injection
Hard	Performance & clean code	inefficient loops
🏆 Reward Function

The environment uses multi-factor scoring:

Issue detection (semantic similarity)
Fix quality (keyword matching)
Concept understanding (embedding similarity)
Severity alignment

Penalties:

Missing issues
Hallucinated issues

Score range: 0.0 → 1.0
Supports partial progress and incremental feedback.

⚙️ Setup
Local
pip install -r requirements.txt
python -m server.app
Docker
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
🌐 API Endpoints
POST /reset → Reset environment, returns initial observation
POST /step → Step with agent action, returns observation, reward, done
📦 Deployment

This environment is deployed as a Docker-based Hugging Face Space, fully reproducible and deterministic.

🧠 Design Choices
Uses local embeddings (SentenceTransformers) for reproducibility
Avoids external API dependencies
Ensures deterministic scoring for evaluation
🏁 Run Example
python inference.py

Example output:

Score: ~0.9+