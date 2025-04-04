# Code Quality & Style

Maintaining consistent code quality and style is crucial for readability, maintainability, and collaboration. This project uses `pre-commit` with `ruff` to automate these checks.

## Tools Used

-   **`ruff`**: An extremely fast Python linter and code formatter, written in Rust. It replaces the need for separate tools like `flake8`, `isort`, `pylint` (partially), and `black`.
-   **`pre-commit`**: A framework for managing and maintaining multi-language pre-commit hooks. It ensures that checks are run automatically before code is committed.

## Configuration (`.pre-commit-config.yaml`)

The `.pre-commit-config.yaml` file defines the hooks that `pre-commit` will run:

-   **Standard Hooks (`pre-commit-hooks`):**
    -   `trailing-whitespace`: Removes trailing whitespace.
    -   `end-of-file-fixer`: Ensures files end with a single newline.
    -   `check-yaml`: Checks YAML files for parseable syntax.
    -   `check-added-large-files`: Prevents accidentally committing large files.
-   **Ruff Hooks (`ruff-pre-commit`):**
    -   `ruff-format`: Formats Python code according to standard conventions (similar to `black`).
    -   `ruff`: Lints Python code for errors, potential bugs, style issues, and unused imports. It attempts to automatically fix issues where possible (`--fix` argument).

## Usage

1.  **Installation:** Ensure you have installed the development dependencies, including `pre-commit`:
    ```bash
    pip install -e .[dev,docs]
    ```
2.  **Install Hooks:** Set up the git hooks in your local repository:
    ```bash
    python -m pre_commit install
    ```
    This only needs to be done once per repository clone.

3.  **Automatic Checks:** Now, every time you run `git commit`, the configured hooks will automatically run against the files you've staged.
    -   If any hooks modify files (e.g., formatting changes), the commit will be aborted. You'll need to `git add` the modified files and run `git commit` again.
    -   If any hooks report errors that cannot be automatically fixed (e.g., syntax errors, undefined variables), the commit will be aborted, and you'll need to fix the issues manually before committing.

4.  **Manual Checks:** You can run the hooks manually on all files at any time:
    ```bash
    python -m pre_commit run --all-files
    ```
    This is useful for checking the entire codebase or after making significant changes.

## Benefits

-   **Consistency:** Ensures all code adheres to the same formatting and style rules.
-   **Early Error Detection:** Catches common errors and potential bugs before they are committed.
-   **Automation:** Reduces the manual effort required for code reviews related to style and basic errors.
-   **Improved Readability:** Consistent formatting makes the code easier to read and understand.
