# Contributing

Thank you for contributing to Text-to-Sketch. All contributions must be made
from a personal fork and submitted to the upstream repository's `dev` branch.
Do not open feature or bug-fix pull requests against `main`.

## 1. Fork and clone the repository

Create a fork on GitHub, then clone your fork. Replace `<your-username>` with
your GitHub username.

```bash
git clone git@github.com:<your-username>/text-to-sketch.git
cd text-to-sketch
```

Your fork should be named `origin`. Add the main repository as `upstream`:

```bash
git remote add upstream \
  git@github.com:naolselemon/text-to-sketch.git
git remote -v
```

If you use HTTPS instead of SSH, use the corresponding HTTPS repository URLs.

## 2. Create a branch from the latest `dev`

Fetch the latest upstream changes and create your working branch directly from
`upstream/dev`:

```bash
git fetch upstream
git switch -c <type>/<short-description> upstream/dev
```

Examples:

```bash
git switch -c feature/add-text-conditioning upstream/dev
git switch -c fix/handle-empty-strokes upstream/dev
git switch -c docs/improve-setup-guide upstream/dev
```

Use a focused, descriptive branch name. Recommended prefixes include
`feature/`, `fix/`, `docs/`, `test/`, and `refactor/`.

## 3. Set up the development environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

See the [README](README.md) for CUDA, data preparation, and model-specific
setup instructions. Do not commit generated datasets, model weights, logs,
credentials, or other local artifacts.

## 4. Make and verify your changes

Keep each pull request limited to one clear purpose. Follow the style and
structure of nearby code, update documentation when behavior changes, and add
or update tests for fixes and new functionality.

Run the test suite before submitting:

```bash
python -B -m unittest discover -s tests -v
```

When your change affects long-sequence or parallel preprocessing behavior, run
the relevant evaluation as well:

```bash
python -B evals/long_sequence_eval.py
python -B evals/parallel_preprocessing_eval.py
```

## 5. Commit and push to your fork

Write a short, imperative commit message that explains the change:

```bash
git add <changed-files>
git commit -m "Add support for ..."
git push -u origin <type>/<short-description>
```

Before opening the pull request, incorporate recent changes from upstream
`dev` and resolve any conflicts locally:

```bash
git fetch upstream
git rebase upstream/dev
git push --force-with-lease
```

Only use `--force-with-lease` on your own feature branch, never on a shared
branch such as `dev` or `main`.

## 6. Open a pull request to `dev`

On GitHub, create a pull request with:

- **base repository:** `naolselemon/text-to-sketch`
- **base branch:** `dev`
- **head repository:** your fork
- **compare branch:** your feature or fix branch

In the pull request description:

- Explain what changed and why.
- Link the related issue, if one exists.
- List the tests or evaluations you ran.
- Mention known limitations or follow-up work.
- Include screenshots, plots, or sample output when they clarify the result.

Before requesting review, confirm that:

- The branch was created from the latest `dev`.
- The pull request targets `dev`, not `main`.
- Tests relevant to the change pass.
- No generated data, large model files, secrets, or unrelated changes are
  included.
- Documentation and configuration examples are updated when needed.

Please respond to review feedback on the same branch; the pull request updates
automatically when you push additional commits to your fork.
