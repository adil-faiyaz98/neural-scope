# Publishing Neural-Scope to PyPI

This guide provides instructions for publishing Neural-Scope to PyPI.

## Prerequisites

Before publishing, ensure you have the following:

1. A PyPI account
2. Twine installed (`pip install twine`)
3. Build tools installed (`pip install build wheel`)
4. All changes committed to the repository

## Preparing for Publication

1. **Update Version Number**

   Update the version number in `advanced_analysis/version.py`:

   ```python
   __version__ = "0.3.0"  # Change this to the new version
   ```

2. **Update RELEASE_NOTES.md**

   Add a new section to `RELEASE_NOTES.md` with the new version number and changes:

   ```markdown
   ## Version 0.3.0 (YYYY-MM-DD)

   ### Major Features
   - Feature 1
   - Feature 2

   ### Improvements
   - Improvement 1
   - Improvement 2

   ### Bug Fixes
   - Bug fix 1
   - Bug fix 2
   ```

3. **Check README_PYPI.md**

   Ensure that `README_PYPI.md` is up-to-date with the latest features and examples.

## Publishing to Test PyPI

It's a good practice to publish to Test PyPI first to ensure everything works correctly.

1. **Run the Publish Script with Test Option**

   ```bash
   python publish.py test
   ```

   This will:
   - Clean build directories
   - Use the PyPI-specific README
   - Build the package
   - Check the package with Twine
   - Publish to Test PyPI

2. **Verify the Package on Test PyPI**

   Visit https://test.pypi.org/project/neural-scope/ to verify that the package was published correctly.

3. **Install from Test PyPI**

   ```bash
   pip install --index-url https://test.pypi.org/simple/ neural-scope
   ```

4. **Test the Installation**

   ```python
   from neural_scope import NeuralScope
   print(NeuralScope.__version__)
   ```

## Publishing to PyPI

Once you've verified that everything works correctly on Test PyPI, you can publish to PyPI.

1. **Run the Publish Script with Prod Option**

   ```bash
   python publish.py prod
   ```

   This will:
   - Clean build directories
   - Use the PyPI-specific README
   - Build the package
   - Check the package with Twine
   - Publish to PyPI

2. **Verify the Package on PyPI**

   Visit https://pypi.org/project/neural-scope/ to verify that the package was published correctly.

3. **Install from PyPI**

   ```bash
   pip install neural-scope
   ```

4. **Test the Installation**

   ```python
   from neural_scope import NeuralScope
   print(NeuralScope.__version__)
   ```

## Automating Publication with GitHub Actions

You can automate the publication process using GitHub Actions.

1. **Create a GitHub Actions Workflow**

   Create a file named `.github/workflows/publish.yml`:

   ```yaml
   name: Publish to PyPI

   on:
     release:
       types: [created]

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: '3.8'
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install build twine wheel
       - name: Build and publish
         env:
           TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
           TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
         run: |
           python publish.py prod
   ```

2. **Add PyPI Credentials to GitHub Secrets**

   - Go to your repository on GitHub
   - Click on "Settings" > "Secrets" > "New repository secret"
   - Add `PYPI_USERNAME` and `PYPI_PASSWORD` with your PyPI credentials

3. **Create a Release**

   - Go to your repository on GitHub
   - Click on "Releases" > "Create a new release"
   - Enter the tag version (e.g., `v0.3.0`)
   - Enter the release title and description
   - Click "Publish release"

   This will trigger the GitHub Actions workflow, which will publish the package to PyPI.

## Troubleshooting

### Common Issues

1. **Version Already Exists**

   If you try to publish a version that already exists on PyPI, you'll get an error. You need to update the version number in `advanced_analysis/version.py`.

2. **README Rendering Issues**

   If your README doesn't render correctly on PyPI, check for syntax errors in your Markdown. You can use the [PyPI Markdown Renderer](https://pypi.org/project/readme-renderer/) to check for issues:

   ```bash
   pip install readme-renderer
   python -m readme_renderer README_PYPI.md
   ```

3. **Missing Dependencies**

   If you get errors about missing dependencies, ensure that all dependencies are listed in `setup.py`.

4. **Authentication Issues**

   If you have authentication issues with PyPI, ensure that your credentials are correct. You can use a `.pypirc` file in your home directory:

   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = your_username
   password = your_password

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = your_username
   password = your_password
   ```

## Conclusion

Publishing Neural-Scope to PyPI makes it easily accessible to users. By following this guide, you can ensure that the package is published correctly and that users can install it with a simple `pip install` command.
