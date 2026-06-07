# Contributing

Agario RL is now a Python-simulator plus browser-game project. Keep changes
small, test-backed, and documented against the current public commands.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .[dev]
```

Frontend setup:

```powershell
cd web
npm install
npm run build
```

## Working conventions

- Preserve deterministic behavior when seeds are fixed.
- Keep Python authoritative for simulation, rewards, observations, PPO, and
  opponent behavior.
- Keep the TypeScript client focused on rendering, interpolation, HUD, and
  input transport.
- Update tests when you change world rules, training behavior, browser frame
  schema, WebSocket input, or public CLI flags.
- Update `README.md`, `AGENTS.md`, and `docs/` when you change workflows,
  telemetry, controls, or config semantics.
- Keep logs, checkpoints, and generated report assets out of routine code
  changes unless the task explicitly asks for a fresh experiment run.

## Checks

Full Python suite:

```powershell
python -m pytest
```

Targeted browser/runtime suite:

```powershell
python -m pytest tests\test_browser_runtime.py tests\test_simulator_upgrade.py tests\test_scenario_training_smoke.py
```

Frontend:

```powershell
cd web
npm run build
```

Docs whitespace:

```powershell
git diff --check -- README.md CONTRIBUTING.md AGENTS.md docs
```

## Pull request expectations

Explain:

- the user-visible runtime or training change
- any config default changes
- test coverage and results
- documentation changes
- whether checkpoints or logs were intentionally generated

If the browser runtime changed, call out WebSocket frame changes, controls, HUD
telemetry, and Browser Use or Playwright smoke-test results.
