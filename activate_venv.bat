@echo off
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Run 'python -m venv .venv' first.
)
