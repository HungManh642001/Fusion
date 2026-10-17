# §9 Error Handling & §10 Logging

## 9.1 Catch at the Right Layer, Never Swallow

Catch an exception only where you can meaningfully recover or translate it. "Meaningfully" means one of: retry with backoff, fall back to a documented alternative, translate into a domain exception with better context, or log + re-raise for a higher layer.

Swallowed exceptions (catch and continue silently, or catch and return a sentinel like `None`) hide bugs and break callers' invariants. If you cannot handle the error at this layer, let it propagate.

## 9.2 Specific Exceptions Only

Never use bare `except:` — it catches `KeyboardInterrupt` and `SystemExit`. Avoid `except Exception:` for control flow; it masks programming errors (`AttributeError`, `TypeError`, `IndexError`) that should crash loudly during development.

Catch the narrowest exception type the operation can actually raise. The message must identify what was being done and which input triggered the failure.

```python
try:
    user = repository.get(user_id)
except DatabaseConnectionError as exc:
    raise UserRepositoryUnavailableError(
        f"Cannot access user repository for user_id={user_id}"
    ) from exc
```

Do not raise bare `Exception("something went wrong")`. Define or use a specific exception subclass so callers can catch the right category.

## 9.3 Preserve Root Cause with `raise ... from`

When translating a low-level exception into a domain exception, chain them with `from exc`. This preserves the full cause chain in the traceback, so operators see both the business-level failure and the technical root cause.

```python
try:
    execute_gpu_workload()
except CUDAOutOfMemoryError as exc:
    raise ResourceExhaustedError(
        "Insufficient GPU memory to run workload."
    ) from exc
```

Use bare `raise` (no `from`) when re-raising the same exception unchanged after logging.

## 9.4 One Stack Trace per Error

Do not log-with-exception and re-raise at every layer in the call stack. Each `logger.exception()` emits a full traceback; three layers doing it produces three near-identical tracebacks that bury the signal.

Pattern: the **innermost** handler that does something (translate, retry, fallback) logs with `logger.exception()`. Higher layers that merely re-raise do **not** log again — they either stay silent or emit a single-line `logger.warning` without the traceback.

```python
def repository():
    try:
        ...
    except DatabaseError:
        logger.exception("Database query failed")  # log here, once
        raise

def service():
    # no try/except — let it propagate silently to the handler above
    repository()

def endpoint():
    # optional single-line warning, no traceback
    logger.warning("Request pipeline aborted")
    service()
```

## 10.1 Logger per Module

Every module creates its own logger named after the module:

```python
import logging
logger = logging.getLogger(__name__)
```

Never use `print()` for runtime diagnostics in production code. `print` goes to stdout unbuffered, bypasses log levels, log rotation, and structured output. It is acceptable only in `if __name__ == "__main__":` entry-point demos.

## 10.2 Log Levels

| Level | Use for |
|-------|---------|
| `DEBUG` | Detailed state dumps, loop iterations, cache hits/misses — off by default in prod |
| `INFO` | Lifecycle events: startup, shutdown, batch completed, model loaded |
| `WARNING` | Recoverable anomalies: retry succeeded, degraded mode, config fallback used |
| `ERROR` | Operation failed and will not be retried; caller must act |
| `CRITICAL` | Process cannot continue; about to exit non-zero |

A timeout that was successfully retried is `WARNING`, not `ERROR`. Promoting every anomaly to `ERROR` desensitizes alerting.

## 10.3 `logger.exception()` in Except Blocks

Inside an `except` block, use `logger.exception(msg)` — it automatically appends the current traceback. Do not use `logger.error(exc)` or `logger.error(str(exc))`; those print only the exception string and lose the stack.

```python
try:
    execute_gpu_workload()
except CUDAOutOfMemoryError:
    logger.exception("GPU workload failed due to out-of-memory.")
    raise
```

## 10.4 Context-Rich Messages

A log line must identify the operation, the entity, and enough context to locate the issue without reading source. Use the `extra=` dict for structured fields that tooling can index.

```python
logger.error(
    "Document indexing failed.",
    extra={"document_id": document_id, "operation": "document_indexing"},
)
```

Forbidden: `logger.error("error")`, `logger.error("request failed")` — no operation, no entity, no cause.

## 10.5 Logging Is Not Handling

Recording an exception does not resolve it. After `logger.exception(...)`, the code must either `raise` (propagate), return a explicitly-typed failure result, or take a documented recovery action. Falling through to the next statement after logging a failure implies success to the caller.

```python
try:
    save_order(order)
except DatabaseError as exc:
    logger.exception("Failed to save order.", extra={"order_id": order.id})
    raise OrderPersistenceError(f"Could not persist order {order.id}") from exc
```
