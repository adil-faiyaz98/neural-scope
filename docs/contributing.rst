Contributing
============

We welcome contributions to Neural-Scope! This document provides guidelines and instructions for contributing.

Setting Up Development Environment
--------------------------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/your-username/neural-scope.git
       cd neural-scope

3. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[all]"
       pip install -r docs/requirements.txt

4. Set up pre-commit hooks:

   .. code-block:: bash

       pip install pre-commit
       pre-commit install

Code Style
---------

We follow PEP 8 style guidelines for Python code. We use flake8 and black for code linting and formatting.

To check your code style:

.. code-block:: bash

    flake8 .

To format your code:

.. code-block:: bash

    black .

Testing
------

We use pytest for testing. To run the tests:

.. code-block:: bash

    pytest

Make sure to write tests for any new features or bug fixes.

Documentation
------------

We use Sphinx for documentation. To build the documentation:

.. code-block:: bash

    cd docs
    make html

The documentation will be built in `docs/_build/html`.

Pull Request Process
------------------

1. Create a new branch for your feature or bug fix:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes and commit them with descriptive commit messages.

3. Push your branch to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

4. Create a pull request from your branch to the main repository.

5. Ensure that all tests pass and the documentation builds successfully.

6. Update the README.md and documentation with details of changes if applicable.

7. The pull request will be merged once it receives approval from maintainers.

Reporting Issues
--------------

If you find a bug or have a feature request, please create an issue on GitHub. Please include:

- A clear and descriptive title
- A detailed description of the issue or feature request
- Steps to reproduce the issue (for bugs)
- Expected behavior
- Actual behavior
- Screenshots or code snippets if applicable
- Environment information (OS, Python version, package versions)

Code of Conduct
-------------

We expect all contributors to follow our Code of Conduct. Please be respectful and considerate of others when contributing to the project.

License
------

By contributing to Neural-Scope, you agree that your contributions will be licensed under the project's MIT License.
