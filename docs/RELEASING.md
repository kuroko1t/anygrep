# Releasing

Releases are published to PyPI by `.github/workflows/release.yml` when a tag `vX.Y.Z` is pushed. It uses
PyPI Trusted Publishing, so there is no API token anywhere in the repository or in GitHub secrets.

## One-time setup (first release only)

Do these in order.

1. Create the GitHub repository `kuroko1t/anygrep` as private, and push `main` only (no other
   branches, no `--all` or `--mirror`):

   ```sh
   git remote add origin git@github.com:kuroko1t/anygrep.git
   git push -u origin main
   ```

2. Check that CI is green (Actions tab, workflow "CI"). The `test` job runs on Linux, macOS and
   Windows with Python 3.10 and 3.13; the `model` job downloads the model (about 540 MB) and runs a real
   index and search on each OS.

3. Make the repository public, then set the About text (for example "Semantic grep for your notes,
   PDFs and images. Local, CPU-only.") and topics such as `semantic-search`, `cli`, `grep`, `embeddings`,
   `onnx`, `local-first`.

4. Create the `pypi` environment on GitHub: Settings -> Environments -> New environment -> `pypi`.
   Add yourself as a required reviewer, so every upload waits for a click.

5. Register the trusted publisher on PyPI. The project does not exist on PyPI yet, so use a "pending
   publisher": log in at https://pypi.org, go to Account settings -> Publishing -> "Add a new pending
   publisher" -> GitHub, and enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `anygrep` |
   | Owner | `kuroko1t` |
   | Repository name | `anygrep` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   Check again just before this step that the name `anygrep` is still free on PyPI
   (https://pypi.org/project/anygrep/ should return 404). A pending publisher does not reserve the name.

For 0.1.0 the version and the CHANGELOG section are already in place; set the date in the CHANGELOG
heading to the release day and go on with step 3 below.

## Every release

1. Make sure `main` is green in CI on all three operating systems.
2. Set the version in `src/anygrep/__init__.py` (`__version__`), and move the `Unreleased` entries in
   `CHANGELOG.md` under a new heading with the version and date. Update the compare links at the bottom.
3. Build and check locally (optional, the workflow does the same):

   ```sh
   python -m pip install build twine
   python -m build --outdir /path/to/scratch/dist
   twine check --strict /path/to/scratch/dist/*
   ```

4. Commit, tag and push:

   ```sh
   git commit -am "Release 0.1.0"
   git tag -a v0.1.0 -m "anygrep 0.1.0"
   git push origin main v0.1.0
   ```

   The workflow refuses to publish if the tag does not match `__version__`.

5. Watch the "Release" workflow. If you added a required reviewer to the `pypi` environment, approve the
   `publish` job. It then uploads to PyPI and creates a GitHub release with the built files.
6. Verify from a clean environment:

   ```sh
   pipx install anygrep==0.1.0 && anygrep --version
   uvx anygrep --help
   ```

A version number can be uploaded to PyPI only once. If something is wrong after the upload, fix it and
release the next patch version; a broken release can be "yanked" on PyPI so that installers skip it.
