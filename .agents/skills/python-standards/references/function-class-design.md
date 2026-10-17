# §6 Function Design & §7 Class Design

## 6.1 Single Responsibility

A function does one thing: either it orchestrates (calls other functions in sequence) or it implements (does the actual work). If a function needs a "and also" in its description, split it.

- Orchestrator: `plan_trajectory()` calls `preprocess()`, `search()`, `smooth()`.
- Implementer: `_expand_node(state)` does one expansion step.

This makes unit testing trivial: test each implementer in isolation; test the orchestrator with mocked implementers.

## 6.2 Guard Clauses / Early Returns

Flatten control flow. Validate preconditions at the top, return or raise immediately, then fall through to the happy path. Target: no nesting deeper than 2 levels of `if`.

```python
def update_node_status(node_id: str, status: str) -> bool:
    if not node_id:
        raise ValueError("node_id is required")
    if status not in ALLOWED_STATUSES:
        return False

    db.update(node_id, status)
    return True
```

The anti-pattern ("arrow code") — nested `if`/`else` that indents the real logic to the right margin — is forbidden.

## 6.3 No Mutable Default Arguments

Defaults are evaluated once at function definition time. A mutable default (`[]`, `{}`, `set()`) is shared across all calls, causing state leakage between invocations.

```python
def calculate_path(waypoints: list[str] | None = None) -> list[str]:
    if waypoints is None:
        waypoints = []
    waypoints.append("End_Node")
    return waypoints
```

## 6.4 Keyword-Only Arguments

Use `*` to force callers to name parameters that are easy to confuse positionally (similar types, optional flags, values with units).

```python
def setup_cluster(host: str, port: int, *, timeout_sec: float = 5.0, retries: int = 3): ...
# Correct:  setup_cluster("127.0.0.1", 8080, timeout_sec=10.0, retries=5)
# Rejected: setup_cluster("127.0.0.1", 8080, 10.0, 5)
```

Apply when: two adjacent params share a type; a param's meaning depends on context (seconds vs milliseconds); there are ≥ 3 optional params.

## 6.5 Function Length

Target ≤ 30 lines of body (excluding docstring). Longer functions almost always violate single responsibility. If a function grows past 30 lines, extract sub-steps into named helpers even before the logic is stable — the names document intent.

## 7.1 Composition over Inheritance

Prefer injecting collaborators (composition) over subclassing for behavior reuse. Inheritance fixes the behavior at class-definition time and creates fragile base-class coupling; composition lets you swap collaborators at runtime and keeps modules decoupled.

```python
class RAGPipeline:
    def __init__(self, vector_db: VectorDBClient, llm: LLMEngine):
        self._db = vector_db
        self._llm = llm
```

Inheritance is appropriate only for true "is-a" relationships where the subclass extends the parent's contract (e.g., `KinodynamicAstar(AStarSearch)`). Multi-level or multi-inheritance hierarchies are a smell — flatten them.

## 7.2 Dataclasses for Data Objects

Any object whose purpose is to carry data (no behavior beyond `__eq__`/`__repr__`) gets `@dataclass`. Add `frozen=True` when the object should be immutable after construction — this is the default for value objects, track records, config snapshots.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RadarTrack:
    track_id: str
    latitude: float
    longitude: float
    confidence_score: float
```

Use `field(default_factory=list)` for mutable defaults inside dataclasses.

## 7.3 Enums for Finite Value Sets

Any closed set of named values (strategies, statuses, modes, directions) gets an `Enum`. This eliminates magic strings/numbers, enables autocompletion, and centralizes the valid set in one place.

```python
from enum import StrEnum

class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"

strategy = ChunkingStrategy.SEMANTIC
```

Use `StrEnum` (Python 3.11+) when the value must serialize as a plain string (JSON, XML, DB). Use `IntEnum` for bit-flag sets. For pure internal use with no serialization, plain `Enum` is fine.

## 7.4 Private by Convention

Prefix `_` on attributes and methods that are implementation details. The public API of a class is what lacks the prefix. This is a contract, not enforcement — document the public surface in the class docstring.
