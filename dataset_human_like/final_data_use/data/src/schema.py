CANONICAL_INTERACTION_COLUMNS = [
    "dataset",
    "user_id",
    "item_id",
    "timestamp",
    "raw_event",
    "event_value",
    "item_text",
    "item_category",
]

CANONICAL_ITEM_COLUMNS = [
    "dataset",
    "item_id",
    "item_text",
    "item_category",
]

PROFILE_COLUMNS = [
    "dataset",
    "user_id",
    "semantic",
    "total_interactions",
    "active_days",
    "session_count",
    "avg_session_len",
    "hour_hist",
    "weekday_weekend_ratio",
    "behavior",
]


def require_columns(record, columns, context):
    missing = [col for col in columns if col not in record]
    if missing:
        raise ValueError("%s missing required columns: %s" % (context, ", ".join(missing)))


def canonical_interaction(dataset, user_id, item_id, timestamp, raw_event, event_value,
                          item_text="", item_category="unknown"):
    return {
        "dataset": str(dataset),
        "user_id": str(user_id),
        "item_id": str(item_id),
        "timestamp": int(float(timestamp or 0)),
        "raw_event": str(raw_event or ""),
        "event_value": float(event_value or 0.0),
        "item_text": str(item_text or ""),
        "item_category": str(item_category or "unknown"),
    }


def canonical_item(dataset, item_id, item_text, item_category="unknown", **extra):
    row = {
        "dataset": str(dataset),
        "item_id": str(item_id),
        "item_text": str(item_text or ""),
        "item_category": str(item_category or "unknown"),
    }
    for key, value in extra.items():
        if value is not None:
            row[key] = value
    return row
