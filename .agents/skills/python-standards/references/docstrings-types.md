# §4 Docstrings & §5 Type Hints

## 4.1 Google-Style Docstrings (mandatory)

Every module, class, and public method gets a docstring in Google style: summary line, blank line, then `Args:`, `Returns:`, `Raises:` sections as applicable. Private helpers (`_method`) may omit the full structure but still need a one-line summary.

```python
def search_similar_chunks(query_vector: list[float], top_k: int = 5) -> list[Document]:
    """Find text chunks with highest cosine similarity to the query vector.

    Args:
        query_vector: Embedding of the user question.
        top_k: Maximum documents returned. Default 5.

    Returns:
        List of Document objects sorted by score descending.

    Raises:
        VectorDBConnectionError: When the Milvus/Qdrant cluster is unreachable.
    """
```

Quy tắc:
- Dòng summary: giọng mệnh lệnh, không dấu chấm cuối câu, ≤ 88 ký tự.
- `Args:` — mỗi tham số một dòng: `tên: mô tả.` Ghi rõ đơn vị, khoảng giá trị, giá trị mặc định.
- `Returns:` — mô tả giá trị trả về; ghi chú tác dụng phụ nếu có.
- `Raises:` — chỉ liệt kê ngoại lệ mà hàm này chủ động ném ra hoặc truyền tiếp từ các phụ thuộc đã tài liệu hóa.
- Không dùng thẻ HTML/XML. Chỉ văn bản thuần.
- Ruff kiểm tra qua `[tool.ruff.lint.pydocstyle] convention = "google"`.

### 4.1.1 Ngôn ngữ: tiếng Việt (bắt buộc)

Mọi docstring và comment trong repo này viết bằng **tiếng Việt**. Ví dụ đúng chuẩn:

```python
def smooth_path(
    path: list[PlannerState],
    origin: Point,
    target: Point,
    *,
    smoothing_cfg: SmoothingSettings | None = None,
) -> list[PlannerState]:
    """Tìm chuỗi con waypoint ngắn nhất khả thi bằng quy hoạch động từ O tới T.

    Args:
        path: Chuỗi waypoint ban đầu gồm các tuple (point, heading).
        origin: Tọa độ điểm xuất phát (O).
        target: Tọa độ mục tiêu (T).
        smoothing_cfg: Section làm mượt DP; mặc định lấy từ ``DEFAULTS.smoothing``.

    Returns:
        Chuỗi waypoint đã làm mượt vẫn thỏa mọi ràng buộc động học.
    """
```

Giữ nguyên không dịch: tên API/code (hàm, lớp, tham số, khóa dict, cờ CLI), hằng số kỹ thuật, đơn vị (`m`, `rad`, `độ`, `giây`), đường dẫn tệp/module, thông báo lỗi do thư viện ba bên ném ra. Các header section của Google-style (`Args:`, `Returns:`, `Raises:`, `Attributes:`) giữ nguyên tiếng Anh — chỉ nội dung mô tả sau dấu hai chấm mới viết bằng tiếng Việt. Không trộn hai ngôn ngữ trong cùng một câu.

Lý do: đội ngũ đọc hiểu tiếng Việt trực tiếp, hạn chế sai nghĩa khi đọc lại code sau nhiều tháng; tên kỹ thuật giữ nguyên để tìm kiếm theo mã nguồn và đối chiếu tài liệu upstream không bị lệch.

## 4.2 TODO / FIXME

- Blank TODOs are forbidden. Every marker carries an owner or ticket ID.
- Format: `# TODO(HungManh-ENG42): add Cross-Encoder reranking next sprint.`
- `FIXME` for known bugs: `# FIXME(ENG42): returns None on empty input, should raise ValueError.`

## 5.1 Built-in Generics (PEP 585)

Use `list`, `dict`, `tuple`, `set` directly. Do not import from `typing`.

```python
def aggregate_metrics(records: list[dict[str, float]]) -> tuple[float, float]: ...
```

Exception: `typing.Literal`, `typing.Protocol`, `typing.TypeVar`, `typing.ClassVar`, `typing.Final` have no built-in equivalent — import those from `typing`.

## 5.2 Union Operator (PEP 604)

`|` replaces `Union` and `Optional`.

```python
def parse_coordinate(value: str | None) -> float | None: ...
```

Do not write `from typing import Optional, Union` in new code.

## 5.3 Protocol for Behavioral Interfaces (PEP 544)

When a function needs a *behavior* (a set of methods) rather than a specific class, define a `Protocol`. Any structurally compatible object satisfies it — no inheritance required. This keeps coupling loose and matches Python's duck typing.

```python
from typing import Protocol

class LLMEngine(Protocol):
    def generate(self, prompt: str) -> str: ...

def process_rag_pipeline(engine: LLMEngine, context: str) -> str:
    return engine.generate(context)
```

Use `Protocol` instead of ABC when: you control neither the implementer nor the consumer; the interface is small (≤ 5 methods); you want to accept third-party objects without modification. Use ABC when: you own all implementers and want runtime enforcement via `register()` or explicit inheritance.

## 5.4 Minimize `Any`

`Any` disables type checking for that expression. Pyright skips all inference downstream of an `Any` operand.

- Use `Mapping[str, object]` for un-parsed JSON payloads.
- Use a concrete dataclass or `TypedDict` once the payload is parsed.
- Reserve `Any` for: plugin entry points, `json.loads()` results before validation, interop with untyped C extensions.
- If you must use `Any`, scope it to a single variable at the boundary and convert to a typed form immediately:

```python
def parse_payload(raw: bytes) -> ParsedPayload:
    data: Mapping[str, object] = json.loads(raw)
    return ParsedPayload(**data)
```

## 5.5 Complete Annotation Coverage

Pyright runs in `strict` mode with `reportMissingParameterType = "error"` and `reportUnknownVariableType = "error"`. Every function parameter, every return value, and every non-obvious local variable assignment gets an annotation. Module-level constants get annotations too.

```python
MAX_EXPANSIONS: int = 100_000
DEFAULT_TIMEOUT_SEC: float = 30.0
```
