# §11 Testing Standards

## 11.1 Test Pyramid

| Layer | Scope | Examples in this repo |
|-------|-------|----------------------|
| Unit | Single function/class, no I/O, no network, no DB | `tests/test_search.py` — heuristic admissibility, successor generation |
| Integration | Two+ modules interacting, or external boundary (file, subprocess) | Running `plan_trajectory()` end-to-end on a fixture scenario |
| E2E | Full user-facing flow through the public API / CLI | `scripts/run_test.py` batch runs (not part of the pytest suite) |

Rules:
- Every core business logic function (algorithm, computation, data transform) has unit tests.
- External boundaries (file I/O, subprocess, network client) have integration tests that verify connection/setup risk.
- E2E scripts exist for manual/CI smoke checks but never replace unit or integration coverage. Do not add an E2E test as the only test for a new function.

## 11.2 AAA Pattern & Naming

Test function name: `test_<function_under_test>_<input_condition>_<expected_outcome>`.

Body structure — three labeled blocks, in this order:

```python
def test_calculate_total_with_valid_items_returns_sum():
    # Arrange
    items = [10, 20, 30]

    # Act
    result = calculate_total(items)

    # Assert
    assert result == 60
```

- **Arrange**: construct inputs, set up fixtures, mock externals. No calls to the function under test.
- **Act**: invoke the function under test exactly once (or the minimal call sequence). Capture the result.
- **Assert**: verify the result. One logical assertion per test; if you need two unrelated asserts, split into two tests.

Naming examples:
- `test_heuristic_with_zero_distance_returns_zero`
- `test_plan_trajectory_with_no_path_raises_planning_error`
- `test_parse_scenario_with_missing_goal_field_raises_value_error`

Forbidden names: `test_it_works`, `test_basic`, `test_calc` — they carry no diagnostic value when the test fails.

## 11.3 Determinism

A test must produce the same pass/fail outcome on every run, on any machine, at any time. Sources of non-determinism to eliminate:

- **Wall clock**: inject a fixed `datetime` via monkeypatch or a parameter. Never call `datetime.now()` inside code under test without a seam for injection.
- **Randomness**: seed all RNGs (`random.seed(42)`, `np.random.default_rng(42)`) or use deterministic algorithms. In this repo, `scenario/generator.py` uses `random` with a fixed seed parameter — tests pass an explicit seed.
- **Filesystem state**: use `tmp_path` fixture; never read/write shared paths.
- **Execution order**: each test sets up its own state in Arrange; no reliance on side effects from other tests.
- **Floating point**: compare with `pytest.approx()` or explicit tolerance, never `==` for floats derived from computation.

## 11.4 Mocking Boundaries

Mock **only** external system boundaries: HTTP clients, database drivers, file-system mocks for I/O-heavy paths, subprocess calls, clock/RNG for determinism.

Do **not** mock internal domain logic. If you find yourself mocking a class defined in the same package as the code under test, the design is coupling too tightly — refactor to composition instead.

Patch at the **use site**, not the definition site:

```python
def test_fetch_user_handles_timeout(mocker):
    # Patch where the code under test looks it up
    mocker.patch("path_planner.scenario.preprocessing.requests.get",
                 side_effect=Timeout)
    result = fetch_user_data(user_id=1)
    assert result == {"error": "service_unavailable"}
```

Using `mocker` (pytest-mock) over raw `unittest.mock.patch` gives automatic cleanup and better failure messages.

## 11.5 Regression Tests for Production Bugs

Before writing a fix for a bug found in production or staging:

1. Write a test that reproduces the bug. Run it. Confirm it **fails**.
2. Write the fix.
3. Re-run the test. Confirm it **passes**.
4. Commit both the test and the fix together.

The test is the safety net: future refactors that reintroduce the bug will fail CI before release. A bug fix without a regression test is incomplete.

## 11.6 Markers

Use the markers declared in `pyproject.toml`:

- `@pytest.mark.unit` — fast, isolated, no I/O (default assumption; mark explicitly for clarity).
- `@pytest.mark.integration` — touches filesystem or multi-module interaction.
- `@pytest.mark.e2e` — full pipeline run.
- `@pytest.mark.slow` — > 5 s runtime; excluded from default CI fast lane via `-m "not slow"`.

Run subsets: `uv run pytest -m unit`, `uv run pytest -m "not slow"`.

## 11.7 Coverage

Branch coverage is enabled (`[tool.coverage.run] branch = true`). Target ≥ 80% on `src/` (enforced by `--cov-fail-under` once baseline is measured). New code should ship with tests that cover both branches of every `if`/`except` it introduces. Omitted from coverage: `__init__.py`, `py.typed`, pure constant files.
