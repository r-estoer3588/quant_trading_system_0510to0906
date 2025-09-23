# Gemini Custom Instructions

This file provides instructions for Gemini to follow when interacting with this project.

## Project Overview

This is a quantitative trading system. The core logic is in the `core` directory, with different trading systems implemented in `core/system*.py`. Strategies for these systems are defined in the `strategies` directory. The `common` directory contains shared utilities for data loading, caching, and broker interactions.

## General Instructions

- When modifying code, please adhere to the existing coding style (PEP 8, flake8).
- Ensure that any new code is covered by tests.
- When adding new dependencies, update `requirements.txt` and `requirements-dev.txt`.
- Use the tools provided to analyze the code before making changes.
- Summarize changes clearly in commit messages.
