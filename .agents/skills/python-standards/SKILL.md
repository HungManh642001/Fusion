---
name: python-standards
description: Mandatory coding standards for this repo. Invoke before writing, editing, or reviewing any Python file under src/, tests/, or scripts/. Covers formatting, naming, docstrings, type hints, function/class design, error handling, logging, testing, imports, configuration, and tooling (ruff, pyright, pytest, bandit, uv).
license: internal
compatibility: antigravity
metadata:
  domain: language
  scope: implementation
  source: docs/Python_Coding_Standards.docx
---

# Python Coding Standards

Mandatory rules derived from `docs/Python_Coding_Standards.docx`. Every Python change in this repo must satisfy all applicable rules below. The 7 quality goals — Readable, Understandable, Testable, Debuggable, Maintainable, Secure, Production-ready — are the acceptance criteria; when a rule is ambiguous, choose the option that best serves those goals.

## Workflow

1. **Load references** — Open the reference files listed below. Load only what the task touches; load §2–§3 for any code edit, §4 for test work, §5 for dependency/packaging changes.
2. **Implement** — Write code that satisfies every applicable rule. When a rule conflicts with existing local style, follow the rule; flag the conflict to the user rather than silently deviating.
3. **Validate** — Run the validation commands (below) and fix every finding before reporting completion. A change is not done until all four pass clean on the touched files.

## Validation Commands

Run from repo root, via `uv run`:

```bash
uv run ruff check --fix <touched files>
uv run ruff format <touched files>
uv run pyright <touched files>
uv run bandit -c pyproject.toml <touched python files>
uv run pytest <touched test files>   # if tests were added/changed
```

All four (lint, format, types, security) must report zero errors. Warnings from `reportUnknownMemberType` / `reportUntypedBaseClass` / `reportUnnecessaryTypeIgnoreComment` are acceptable per `pyproject.toml`; everything else is an error to fix.

## Reference Files

| File | Contents | Load when |
|------|----------|-----------|
| `references/formatting-naming.md` | Indentation, line length, comments, trailing commas, naming conventions, import ordering | Any code edit |
| `references/docstrings-types.md` | Google-style docstrings, TODO/FIXME, type hints (built-in generics, PEP 604 union, Protocol, avoid Any) | Any public API, class, or module |
| `references/function-class-design.md` | Single responsibility, guard clauses, no mutable defaults, keyword-only args, composition over inheritance, dataclasses, enums | Defining functions or classes |
| `references/error-logging.md` | Exception handling, `raise ... from`, no swallowing, logger per module, log levels, context-rich logs, no log-as-handling | Any try/except or logging |
| `references/testing.md` | Test pyramid, AAA pattern + naming, determinism, mocking boundaries, regression tests | Writing or modifying tests |
| `references/packaging-config.md` | pyproject.toml, src layout, venvs, dependency groups, lockfiles, no hardcoded config, no secrets in code, immutable settings | Adding deps, new modules, config, or environment changes |

## Hard Rules (always apply, all file types)

These are the highest-leverage rules; they catch most violations even without loading a reference file:

1. **4-space indent, 88-char lines, double quotes, trailing comma on multiline collections.**
2. **Full type annotations** on every function parameter and return value; use `list[str]` not `List[str]`, `X | None` not `Optional[X]`; `Protocol` for behavioral interfaces; `Any` only for un-parsed dynamic JSON/plugin data.
3. **Google-style docstring** on every module, class, and public method with `Args:`, `Returns:`, `Raises:` sections where applicable. All docstrings and comments are written in **Vietnamese** (see rule 15).
4. **No mutable default arguments** (`= []`, `= {}`); use `None` sentinel.
5. **Guard clauses / early returns** to keep nesting ≤ 2 levels; no arrow code.
6. **Specific exceptions only** — never bare `except:` or `except Exception:` for logic control flow; always `raise ... from exc` when re-raising as a different type.
7. **`logger = logging.getLogger(__name__)`** per module; `logger.exception()` inside `except` blocks (never `logger.error(exc)`); log messages carry operation context via `extra={...}`; no `print()` for runtime info.
8. **Imports**: standard library → third-party → first-party (`path_planner`), two blank lines after imports; absolute imports only; no wildcard `from x import *`; no imports inside functions.
9. **No hardcoded secrets, URLs, timeouts, or environment-dependent values** in code; inject via a frozen `@dataclass Settings` loaded from env.
10. **Tests**: name `test_<fn>_<condition>_<expected>`; body split into Arrange / Act / Assert; deterministic (no wall clock, no RNG without fixed seed); mock only external boundaries at the call site; every production bug gets a failing regression test before the fix.
11. **Composition over inheritance**; `@dataclass(frozen=True)` for immutable data objects; `StrEnum` for finite value sets.
12. **Keyword-only args** (`*`) for parameters that are easy to confuse positionally.
13. **TODO/FIXME** must carry an owner or ticket ID: `# TODO(HungManh-ENG42): ...`. Blank TODOs are forbidden.
14. **One concern per function**; if a function orchestrates, delegate each step to a single-responsibility helper.
15. **Tiếng Việt cho mọi docstring và comment.** Docstring mô tả (summary line, phần `Args:`/`Returns:`/`Raises:`, body) và comment dùng tiếng Việt; giữ nguyên không dịch: tên API/code (hàm, lớp, tham số, khóa dict, cờ CLI), hằng số kỹ thuật, đơn vị (`m`, `rad`, `độ`, `giây`), đường dẫn tệp/module, thông báo lỗi do thư viện ba bên ném ra. Cấu trúc section của Google-style vẫn giữ nguyên từ tiếng Anh (`Args:`, `Returns:`, `Raises:`, `Attributes:`) — chỉ nội dung mô tả mới viết bằng tiếng Việt. Không để hai ngôn ngữ trộn lẫn trong cùng một câu.

## Tooling Ground Truth

The authoritative tool configuration lives in `pyproject.toml` — do not restate it here. The tools and their roles:

- **ruff** — linter + formatter (replaces flake8/black/isort). Config: `[tool.ruff]`.
- **pyright** — strict static type checker. Config: `[tool.pyright]`, `typeCheckingMode = "strict"`.
- **pytest** — test runner with coverage gate. Config: `[tool.pytest.ini_options]`.
- **bandit** — security scanner. Config: `[tool.bandit]`.
- **uv** — env + dependency manager; lockfile `uv.lock` is the reproducibility mechanism.
- **pre-commit** — runs ruff + pyright + bandit on every commit (see `.pre-commit-config.yaml`).

If a rule in this skill conflicts with `pyproject.toml`, the config wins for tool behavior; this skill wins for code-authoring judgment the tools cannot express.
