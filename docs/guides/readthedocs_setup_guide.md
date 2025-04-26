# Setting Up Read the Docs for Neural-Scope

This guide will walk you through the process of setting up Read the Docs for the Neural-Scope documentation.

## Prerequisites

- A GitHub account with the Neural-Scope repository
- A Read the Docs account

## Steps to Set Up Read the Docs

### 1. Create a Read the Docs Account

1. Go to [Read the Docs](https://readthedocs.org/) and click on "Sign Up"
2. You can sign up using your GitHub account for easier integration

### 2. Connect Your GitHub Account

1. After signing in, go to your Account Settings
2. Click on "Connected Services"
3. Connect your GitHub account if it's not already connected

### 3. Import Your Repository

1. Go to your Read the Docs dashboard
2. Click on "Import a Project"
3. Select "Import Manually" if your repository isn't listed
4. Fill in the following details:
   - **Name**: neural-scope
   - **Repository URL**: https://github.com/adil-faiyaz98/neural-scope.git
   - **Repository type**: Git
   - **Default branch**: main

### 4. Configure Build Settings

1. After importing, go to the project's "Admin" section
2. Click on "Advanced Settings"
3. Make sure the following settings are configured:
   - **Documentation type**: Sphinx
   - **Python configuration file**: docs/conf.py
   - **Python interpreter**: CPython 3.x
   - **Requirements file**: docs/requirements.txt

### 5. Trigger a Build

1. Go to the "Builds" section
2. Click on "Build Version"
3. Select the main branch and click "Build"
4. Monitor the build process for any errors

### 6. Set Up Custom Domain (Optional)

If you want to use a custom domain (e.g., docs.neural-scope.com):

1. Go to the project's "Admin" section
2. Click on "Domains"
3. Add your custom domain
4. Configure your DNS settings as instructed

### 7. Verify the Documentation

1. Once the build is complete, visit your documentation at https://neural-scope.readthedocs.io/
2. Verify that all pages are rendering correctly
3. Check for any missing images or broken links

## Troubleshooting Common Issues

### Build Failures

If your build fails, check the build logs for specific errors. Common issues include:

- Missing dependencies in requirements.txt
- Syntax errors in RST files
- Missing referenced files

### Missing Images

If images are not displaying:

1. Make sure they are properly referenced in your documentation
2. Check that the paths are correct
3. Ensure the images are committed to the repository

### Slow Builds

If builds are taking too long:

1. Optimize your documentation structure
2. Reduce the number of dependencies
3. Consider using Read the Docs' build cache

## Maintaining Your Documentation

### Automatic Builds

Read the Docs automatically builds your documentation when you push changes to your repository. To ensure smooth builds:

1. Test your documentation locally before pushing
2. Use a consistent formatting style
3. Keep your dependencies up to date

### Versioning

You can create different versions of your documentation:

1. Go to the "Versions" section in your project
2. Activate the versions you want to build
3. Configure version-specific settings if needed

## Additional Resources

- [Read the Docs Documentation](https://docs.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
