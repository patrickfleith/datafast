# Publishing Guide for datafast

This document explains how the automated version bumping and publishing process works for the datafast package.

## Automated Workflow

The package uses GitHub Actions to automatically:

1. Bump the version number in `pyproject.toml`
2. Build the package
3. Tag the branch with the new version
4. Publish to PyPI

This process is triggered whenever:
- A push is made to the `main` branch (typically through a merge)
- The workflow is manually dispatched through GitHub Actions interface

## Required Secrets

To enable PyPI publishing, you need to set up the following secrets in your GitHub repository:

1. `PYPI_USERNAME`: Your PyPI username
2. `PYPI_PASSWORD`: Your PyPI password or token (recommended)

### Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings"
3. Click on "Secrets and variables" then "Actions"
4. Click "New repository secret"
5. Add the required secrets

## Version Bumping Strategy

The workflow intelligently handles version bumping based on whether you've made manual changes:

### Automatic Patch Version Increments

If you haven't manually changed the version in `pyproject.toml`, the workflow automatically increments the patch version (the third number in the semantic versioning scheme).

For example:
- 0.0.9 → 0.0.10
- 1.2.3 → 1.2.4

### Manual Version Changes

If you manually update the version in `pyproject.toml` before merging to main, the workflow will detect this change and preserve it without further incrementing:

- If you change 0.0.9 to 0.1.0 manually, it will remain 0.1.0
- If you change 0.0.9 to 1.0.0 manually, it will remain 1.0.0

This allows you to make major or minor version bumps as needed while still using the automated workflow.

## Manual Publishing

If you need to trigger the publishing process manually:

1. Go to your GitHub repository
2. Click on "Actions"
3. Select the "Publish Python Package" workflow
4. Click "Run workflow"
5. Select the branch and click "Run workflow"
