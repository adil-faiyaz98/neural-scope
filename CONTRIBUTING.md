# Contributing to Advanced Analysis

Thank you for your interest in contributing to Advanced Analysis! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## How to Contribute

There are many ways to contribute to Advanced Analysis:

1. **Report bugs**: If you find a bug, please create an issue with a detailed description of the problem, steps to reproduce, and your environment.
2. **Suggest features**: If you have an idea for a new feature or enhancement, please create an issue to discuss it.
3. **Improve documentation**: Help us improve our documentation by fixing typos, adding examples, or clarifying explanations.
4. **Submit code changes**: Fix bugs, add features, or improve existing code by submitting pull requests.

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/advanced-analysis.git
   cd advanced-analysis
   ```
3. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature-or-bugfix-name
   ```

## Code Style

We follow the [Black](https://black.readthedocs.io/) code style and use [isort](https://pycqa.github.io/isort/) for import sorting. You can format your code with:

```bash
black advanced_analysis tests
isort advanced_analysis tests
```

We also use [flake8](https://flake8.pycqa.org/) for linting and [mypy](http://mypy-lang.org/) for type checking:

```bash
flake8 advanced_analysis tests
mypy advanced_analysis
```

## Testing

Please add tests for any new features or bug fixes. We use [pytest](https://docs.pytest.org/) for testing:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=advanced_analysis
```

## Pull Request Process

1. Update the documentation (docstrings, README, etc.) with details of changes.
2. Add or update tests to reflect your changes.
3. Ensure all tests pass and code style checks pass.
4. Update the CHANGELOG.md with a description of your changes.
5. Submit a pull request to the `main` branch.

## Release Process

1. Update the version number in `pyproject.toml` and `setup.py` following [Semantic Versioning](https://semver.org/).
2. Update the CHANGELOG.md with the new version and release notes.
3. Create a new release on GitHub with a tag matching the version number.

## License

By contributing to Advanced Analysis, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
