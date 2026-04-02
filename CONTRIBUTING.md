# Contributing

Thanks for contributing to Agario RL observer cockpit. The project favors small
test-backed changes, clear runtime behavior, and documentation that matches the
current code.

## Development setup

Start in the project root and create a local environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Working conventions

Keep these rules in mind while you work:

- keep behavior changes scoped and focused
- preserve deterministic behavior when seeds are fixed
- update tests when you change world rules, trainer behavior, controller
  semantics, or render snapshots
- update `README.md` and the pages in `docs/` when you change controls,
  telemetry, or config semantics
- prefer semantic supervisor commands over renderer-specific input plumbing
- keep rendering isolated from RL logic by building immutable frame snapshots

## Run checks before opening a PR

Run the full suite before you send a change for review.

```bash
python -m pytest -q
```

When you work on the observer cockpit, this targeted subset usually catches the
important regressions faster:

```bash
python -m pytest -q ^
  tests/test_controller_ui_toggles.py ^
  tests/test_render_frame_snapshot.py ^
  tests/test_render_backend_factory.py ^
  tests/test_render_backend_integration.py ^
  tests/test_render_backend_compat.py ^
  tests/test_supervisor_runtime_stats.py
```

## Pull request expectations

Use the PR description to explain:

- the user-visible runtime change
- any config default changes
- test coverage and results
- any documentation changes

If a change affects the cockpit, call out:

- new or removed controls
- telemetry changes
- compatibility behavior for old backend names

## Reporting issues

When you file an issue, include:

- operating system and Python version
- the command you ran
- any config overrides
- the full traceback or exact reproduction steps
- whether you used the default Raylib path or a legacy backend alias
