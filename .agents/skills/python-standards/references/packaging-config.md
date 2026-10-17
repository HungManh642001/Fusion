# §12 Packaging & Dependencies, §13 Configuration, §14 Tooling

## 12.1 Single Source of Truth: `pyproject.toml`

All project metadata, dependencies, and tool configuration live in `pyproject.toml`. There is no `setup.py`, no `requirements.txt` for project deps (those are for ad-hoc scripts only). The `[build-system]` block specifies the build backend (`uv_build`).

When adding a dependency, edit `pyproject.toml` under the correct group, then run `uv lock` to update `uv.lock`. Never hand-edit `uv.lock`.

## 12.2 src Layout

Source code lives under `src/path_planner/`. Tests under `tests/`. Scripts under `scripts/`. This layout prevents Python from accidentally importing local source instead of the installed package during testing. Do not create top-level packages alongside `pyproject.toml`.

## 12.3 Virtual Environments & Dependency Groups

- All work happens inside the `uv`-managed `.venv/`. Never install into the system Python.
- Run every command via `uv run` to guarantee the correct environment: `uv run pytest`, `uv run ruff check .`, etc.
- Dependency groups in `pyproject.toml`:
  - `[project.dependencies]` — runtime only; what Production needs.
  - `[dependency-groups].dev` — lint/type/security tools + includes `test` and `scripts`.
  - `[dependency-groups].test` — pytest and plugins.
  - `[dependency-groups].scripts` — pandas, seaborn for analysis scripts.
- Production deploys use `uv sync --no-dev` (runtime only) or `uv sync --frozen` (lockfile-exact, error on drift).

## 12.4 Reproducibility

`uv.lock` records exact versions and hashes for every transitive dependency. It is committed to git. Production installs must use the lockfile:

```bash
uv sync --frozen        # fail if pyproject.toml drifted from lock
uv sync --no-dev        # production: runtime deps only
```

Loose installs (`pip install path_planner`, `uv add` without locking) are forbidden in CI/CD and deployment pipelines.

## 13.1 No Hardcoded Configuration

Values that vary by environment or deployment go into configuration, never into logic code. This includes but is not limited to: database URLs, service endpoints, timeouts, retry counts, feature flags, model names, batch sizes, log levels, worker counts.

```python
def create_http_client(settings: Settings) -> HttpClient:
    return HttpClient(
        base_url=settings.service_url,
        timeout=settings.request_timeout_seconds,
    )
```

Algorithm-specific constants that are truly invariant across all environments (e.g., `GRAVITY = 9.81`, geometric tolerances for the VCM-06 corridor) may live in `src/path_planner/config.py` as module-level `UPPER_SNAKE_CASE` constants. The test: "would this value differ between dev/staging/prod?" If yes → Settings. If no → config constant.

## 13.2 No Secrets in Code or Git

Passwords, API keys, tokens, private keys, DB credentials, cloud credentials, JWT secrets, encryption keys — none of these appear in source files, test fixtures, or committed config.

- Secrets are injected at runtime via environment variables or a secret manager.
- `.env`, `credentials.json`, `service-account.json`, `*.pem`, `*.key` are in `.gitignore`.
- A `.env.example` with placeholder values documents required variables without exposing real ones.
- Test fixtures may use obviously-fake values (`"test-key-do-not-use"`) — bandit rules S105/S106 are suppressed in `tests/**` for this reason.

## 13.3 Immutable Settings

Configuration is loaded once at startup and treated as immutable thereafter. Use `@dataclass(frozen=True)` for the settings object. Mutating settings at runtime (`settings.max_retries = 100`) is forbidden — it creates invisible state divergence between components.

```python
@dataclass(frozen=True)
class Settings:
    database_url: str
    max_retries: int
    request_timeout_sec: float
```

## 14.1 Ruff

Linter + formatter. Config in `[tool.ruff]`. Key settings already in `pyproject.toml`:
- `line-length = 88`, `indent-width = 4`, `target-version = "py313"`
- Rule sets: E, W, F, I, N, D, UP, B, SIM, S, ASYNC, RUF
- Pydocstyle convention: `google`
- Per-file ignores for `tests/**` (S101 assert, D docstrings, S105/S106 test secrets)

Commands: `uv run ruff check --fix .`, `uv run ruff format .`.

## 14.2 Pyright

Strict static type checker. Config in `[tool.pyright]`: `typeCheckingMode = "strict"`, `pythonVersion = "3.13"`. Missing parameter types, unknown variable types, and unknown argument types are **errors**. `reportUnknownMemberType`, `reportUntypedBaseClass`, `reportUnnecessaryTypeIgnoreComment` are warnings (acceptable for third-party interop).

Command: `uv run pyright`. Zero errors required before commit.

## 14.3 Pytest

Config in `[tool.pytest.ini_options]`. Markers: `unit`, `integration`, `e2e`, `slow`. Coverage runs by default on `src/` with branch coverage. `filterwarnings = ["error"]` promotes unhandled warnings to failures (DeprecationWarning excluded).

Commands: `uv run pytest`, `uv run pytest -m unit`, `uv run pytest tests/test_search.py::test_heuristic_admissible -v`.

## 14.4 Bandit & pip-audit

- **bandit** scans source for security vulnerabilities (hardcoded passwords, `eval()`, SQL injection patterns, weak crypto). Config: `[tool.bandit]`, skips B101 (assert). Command: `uv run bandit -c pyproject.toml -r src/ -q`.
- **pip-audit** checks installed dependencies against published CVEs. Command: `uv run pip-audit`. Run after any dependency change.

## 14.5 uv

Environment and dependency manager. Key commands:

| Command | Purpose |
|---------|---------|
| `uv sync` | Create/update `.venv` from lockfile (includes dev groups) |
| `uv sync --no-dev` | Production install (runtime deps only) |
| `uv sync --frozen` | Lockfile-exact; error if pyproject.toml changed without re-lock |
| `uv add <pkg>` | Add runtime dep + update lock |
| `uv add --group test <pkg>` | Add to a specific dependency group |
| `uv lock` | Re-resolve and update `uv.lock` after manual pyproject edits |

PyPI index is the internal mirror: `http://192.168.105.40:8510/repository/pypi/simple` (set in `[tool.uv] index-url`).

## 14.6 Pre-commit

Local hooks (see `.pre-commit-config.yaml`) run on every `git commit`:
1. `ruff check --fix` — auto-fix lint issues on staged Python files.
2. `ruff format` — normalize formatting.
3. `pyright` — strict type check (whole project).
4. `bandit` — security scan on `src/`.

Install once: `pre-commit install`. Manual full run: `pre-commit run --all-files`. A commit is blocked if any hook reports errors.

## Validation Checklist (run before declaring work complete)

```bash
uv run ruff check --fix <files> && uv run ruff format <files>
uv run pyright <files>
uv run bandit -c pyproject.toml <python files>
uv run pytest <relevant tests>
```

All four must exit 0. If a new dependency was added, also run `uv run pip-audit`.
