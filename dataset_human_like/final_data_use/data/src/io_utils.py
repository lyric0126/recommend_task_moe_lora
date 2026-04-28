import datetime as _dt
import json
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def root_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_config(path=None):
    path = path or root_path("configs", "pseudo_user_pipeline.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the pipeline config")
    return yaml.safe_load(text)


def stage_paths(stage):
    return (
        root_path("logs", "stage_%s.log" % stage),
        root_path("reports", "checkpoint_stage_%s.md" % stage),
    )


def append_log(stage, message):
    log_path, _ = stage_paths(stage)
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message).rstrip() + "\n")


def write_stage_log(stage, objective, command, summary, files, smoke):
    log_path, checkpoint_path = stage_paths(stage)
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(os.path.dirname(checkpoint_path))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend(["- %s" % p for p in files])
    body.append("Smoke test result: %s" % smoke)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = [
        "# Checkpoint Stage %s" % stage,
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Execution Commands",
        "",
        "`%s`" % command,
        "",
        "## Output Summary",
        "",
        summary,
        "",
        "## Generated Files",
        "",
    ]
    md.extend(["- `%s`" % p for p in files])
    md.extend(["", "## Smoke Test Result", "", smoke, ""])
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def write_v2_stage_log(stage, objective, command, summary, files, smoke):
    log_path = root_path("logs", "stage_v2_%s.log" % stage)
    checkpoint_path = root_path("reports", "checkpoint_v2_stage_%s.md" % stage)
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(os.path.dirname(checkpoint_path))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend(["- %s" % p for p in files])
    body.append("Smoke test result: %s" % smoke)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = [
        "# Checkpoint V2 Stage %s" % stage,
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Execution Commands",
        "",
        "`%s`" % command,
        "",
        "## Output Summary",
        "",
        summary,
        "",
        "## Generated Files",
        "",
    ]
    md.extend(["- `%s`" % p for p in files])
    md.extend(["", "## Smoke Test Result", "", smoke, ""])
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def write_v2fix_stage_log(stage, objective, command, summary, files, smoke, error_summary=""):
    log_path = root_path("logs", "stage_v2fix_%s.log" % stage)
    checkpoint_path = root_path("reports", "checkpoint_v2fix_stage_%s.md" % stage)
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(os.path.dirname(checkpoint_path))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend(["- %s" % p for p in files])
    body.append("Smoke test result: %s" % smoke)
    body.append("Error summary: %s" % (error_summary or "none"))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = [
        "# Checkpoint V2fix Stage %s" % stage,
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Execution Commands",
        "",
        "`%s`" % command,
        "",
        "## Output Summary",
        "",
        summary,
        "",
        "## Generated Files",
        "",
    ]
    md.extend(["- `%s`" % p for p in files])
    md.extend(["", "## Smoke Test Result", "", smoke, "", "## Error Summary", "", error_summary or "none", ""])
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def write_v3_stage_log(stage, objective, command, summary, files, smoke, error_summary=""):
    log_path = root_path("logs", "stage_v3_%s.log" % stage)
    checkpoint_path = root_path("reports", "checkpoint_v3_stage_%s.md" % stage)
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(os.path.dirname(checkpoint_path))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend(["- %s" % p for p in files])
    body.append("Smoke test result: %s" % smoke)
    body.append("Error summary: %s" % (error_summary or "none"))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = [
        "# Checkpoint V3 Stage %s" % stage,
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Execution Commands",
        "",
        "`%s`" % command,
        "",
        "## Output Summary",
        "",
        summary,
        "",
        "## Generated Files",
        "",
    ]
    md.extend(["- `%s`" % p for p in files])
    md.extend(["", "## Smoke Test Result", "", smoke, "", "## Error Summary", "", error_summary or "none", ""])
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parquet_available():
    return pa is not None and pq is not None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_jsonl_fallback(path, rows, schema=None, reason=None):
    ensure_dir(os.path.dirname(path))
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        meta = {
            "__meta__": {
                "storage_format": "jsonl_fallback",
                "requested_path": path,
                "fallback_reason": reason or "parquet backend unavailable or write failed",
                "created_at": _dt.datetime.utcnow().isoformat() + "Z",
                "schema": schema or [],
                "format_note": "This file uses one JSON object per line after this metadata header although the extension may be .parquet.",
            }
        }
        f.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_table(path, rows, schema=None, storage_format="auto"):
    rows = list(rows)
    requested = storage_format or "auto"
    if requested in ("auto", "parquet") and parquet_available():
        try:
            ensure_dir(os.path.dirname(path))
            table = pa.Table.from_pylist([_json_safe(r) for r in rows])
            metadata = {
                b"storage_format": b"parquet",
                b"created_at": (_dt.datetime.utcnow().isoformat() + "Z").encode("utf-8"),
                b"schema_hint": json.dumps(schema or []).encode("utf-8"),
            }
            table = table.replace_schema_metadata(metadata)
            pq.write_table(table, path)
            return len(rows)
        except Exception as exc:
            if requested == "parquet":
                raise
            return _write_jsonl_fallback(path, rows, schema, "parquet write failed: %s" % exc)
    return _write_jsonl_fallback(path, rows, schema, "parquet backend unavailable")


def _looks_like_parquet(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PAR1"
    except IOError:
        return False


def read_table(path, limit=None):
    if _looks_like_parquet(path):
        if not parquet_available():
            raise RuntimeError("File is true parquet but pyarrow is unavailable: %s" % path)
        table = pq.read_table(path)
        rows = table.to_pylist()
        return rows[:limit] if limit else rows
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "__meta__" in obj:
                continue
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def table_storage_info(path):
    info = {"path": path, "exists": os.path.exists(path), "storage_format": "missing", "schema": [], "sample_count": 0}
    if not info["exists"]:
        return info
    if _looks_like_parquet(path):
        info["storage_format"] = "parquet"
        if parquet_available():
            table = pq.read_table(path)
            info["sample_count"] = table.num_rows
            info["schema"] = table.schema.names
        return info
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    try:
        obj = json.loads(first)
        meta = obj.get("__meta__", {})
        info["storage_format"] = meta.get("storage_format", "jsonl_unknown")
        info["schema"] = meta.get("schema", [])
    except Exception:
        info["storage_format"] = "unknown_text"
    try:
        info["sample_count"] = len(read_table(path))
    except Exception:
        info["sample_count"] = 0
    return info


def table_exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def write_markdown(path, title, sections):
    ensure_dir(os.path.dirname(path))
    lines = ["# %s" % title, ""]
    for name, content in sections:
        lines.extend(["## %s" % name, ""])
        if isinstance(content, list):
            lines.extend(content)
        else:
            lines.append(str(content))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_time(value):
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(float(value))
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        pass
    formats = [
        "%a %b %d %H:%M:%S %z %Y",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return int(_dt.datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    raise ValueError("Unrecognized timestamp: %s" % text)


def clean_text(value):
    return " ".join(str(value or "").replace("\n", " ").replace("\t", " ").split())


def sample_rows(rows, n=3):
    return rows[:n]
