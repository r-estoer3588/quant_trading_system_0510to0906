# Gemini Custom Instructions (common)

This directory contains shared utilities and modules used across the entire application.

- Modules in this directory provide services like data loading (`data_loader.py`), caching (`cache_manager.py`), and broker interactions (`broker_alpaca.py`).
- Changes to these modules can have wide-ranging effects. Ensure that any modifications are backward-compatible or that all dependent modules are updated accordingly.
