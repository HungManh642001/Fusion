# §2 Formatting & §3 Naming

## 2.1 Indentation

- 4 spaces per level. No tabs.
- Rationale: consistent layout across editors; avoids Python's silent indent bugs.

```python
def calculate_kinematic_constraints(velocity: float, acceleration: float) -> float:
    return velocity + acceleration * 0.5  # 4 spaces
```

## 2.2 Line Length

- Max 88 characters per line (ruff/black default).
- Wrap long calls by breaking after `(` and before `)`; one argument per line when wrapping.

```python
engine_response = vllm_client.generate(
    prompt=system_prompt,
    max_tokens=2048,
    temperature=0.7,
    timeout=global_timeout_seconds,
)
```

## 2.3 Comments

- Viết bằng **tiếng Việt** (xem §4.1.1 trong docstrings-types.md).
- Giải thích **vì sao** hoặc bổ sung ngữ cảnh mà code không tự diễn đạt được. Không nhắc lại điều code đang làm.
- Comment cũ hơn code còn tệ hơn là không có comment — nếu comment chỉ lặp lại nội dung code, hãy xóa nó.

```python
# Good: explains why
@njit(fastmath=True)  # nopython mode drops GIL for large matrix loops
def optimize_trajectory_matrix(matrix: np.ndarray) -> np.ndarray: ...

# Bad: restates the code
# Function to optimize trajectory
def optimize_trajectory_matrix(matrix):
    return matrix
```

## 2.4 Trailing Commas

- Always add a trailing comma on the last element of multiline lists, dicts, tuples, and call arguments.
- Keeps git diffs minimal: adding an element is one added line, not one modified + one added.

```python
allowed_roles = [
    "admin",
    "reviewer",
]
```

## 3. Naming Conventions

Principles, in priority order: descriptive → consistent → clear → reasonably short.

| Object | Style | Example |
|--------|-------|---------|
| Module / package | `snake_case` | `path_planner`, `search_engine.py` |
| Class | `PascalCase` | `KinodynamicAstar`, `SearchStats` |
| Function / method / variable | `snake_case` | `plan_trajectory`, `max_expansions` |
| Constant (module-level, immutable) | `UPPER_SNAKE_CASE` | `ALLOWED_STATUSES`, `DEFAULT_TIMEOUT_SEC` |
| Private attribute/method | leading `_` | `self._db`, `_expand_node` |
| Type alias / Protocol | `PascalCase` | `LLMEngine`, `Point2D` |
| Test function | `test_<fn>_<condition>_<expected>` | see testing.md |

Forbidden:
- Hungarian notation (`strName`, `iCount`) — type hints replace it.
- Single-letter names except loop indices (`i`, `j`, `k`) and conventional math symbols in tight numeric loops.
- Abbreviations that require a legend (`calc` → `calculate`, `cfg` → `config`, `mgr` → `manager`).

Consistency beats cleverness: if the codebase already uses `waypoint`, do not introduce `wp` in a new module.

## 8. Imports & Modules

Ordering — three blocks separated by one blank line, two blank lines after the last import before code:

```python
import json
import logging

import numpy as np
from fastapi import FastAPI

from path_planner.config import settings
```

Rules:
- Standard library first, then third-party, then first-party (`path_planner`, `RD`).
- Absolute imports from project root. No relative imports deeper than one level (no `from ..core import x`).
- No wildcard imports (`from package import *`) — they hide the namespace and break static analysis.
- No imports inside functions — they slow repeated calls, hide dependencies, and defer import-time errors to runtime. The only acceptable exception is a circular-import workaround, which must be marked with a `# noqa: I252` comment and a one-line justification.
- `from package import Client, RequestError` — import specific names, not the whole module, unless you actually use multiple submodule attributes via dot access.
