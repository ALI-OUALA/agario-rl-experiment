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
- update `AGENTS.md` when you change the project map, safe commands, protected
  artifacts, or the first files future agents must inspect
- prefer semantic supervisor commands over renderer-specific input plumbing
- keep rendering isolated from RL logic by building immutable frame snapshots
- keep logs, checkpoints, and generated report assets out of routine code
  changes unless the task explicitly asks for a fresh experiment run

## Run checks before opening a PR

Run the full suite before you send a change for review.

```bash
python -m pytest -q
```

When you work on the observer cockpit, this targeted subset usually catches the
important regressions faster:

```bash
python -m pytest -q ^
  tests/test_simulator_upgrade.py ^
  tests/test_scenario_training_smoke.py ^
  tests/test_controller_ui_toggles.py ^
  tests/test_render_frame_snapshot.py ^
  tests/test_render_ui_flags.py ^
  tests/test_render_backend_factory.py ^
  tests/test_render_backend_integration.py ^
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
- scenario preset or render-detail changes
- compatibility behavior for old backend names

If a change affects the simulator upgrade, call out:

- new config defaults
- observation shape changes
- reward breakdown changes
- checkpoint compatibility expectations

## Reporting issues

When you file an issue, include:

- operating system and Python version
- the command you ran
- any config overrides
- the full traceback or exact reproduction steps
- whether you used the default Raylib path or a legacy backend alias
