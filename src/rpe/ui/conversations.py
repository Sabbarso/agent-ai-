"""Persistent chat history: one JSON file per conversation in `data/conversations/`."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

TITLE_CHARS = 60


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_title(question: str) -> str:
    q = " ".join(question.split())
    return q if len(q) <= TITLE_CHARS else q[: TITLE_CHARS - 1].rsplit(" ", 1)[0] + "…"


class ConversationStore:
    """Create, list, load, save and delete conversations (atomic JSON writes)."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _path(self, conv_id: str) -> Path:
        return self.dir / f"{conv_id}.json"

    def new(self) -> dict[str, Any]:
        now = _now()
        return {
            "id": uuid.uuid4().hex[:12],
            "title": "Nouvelle conversation",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    def save(self, conv: dict[str, Any]) -> None:
        """Persist a conversation (empty conversations are not written)."""
        if not conv["messages"]:
            return
        if conv["title"] == "Nouvelle conversation":
            first = next((m for m in conv["messages"] if m["role"] == "user"), None)
            if first:
                conv["title"] = make_title(first["content"])
        conv["updated_at"] = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(conv["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(conv["id"]))

    def load(self, conv_id: str) -> dict[str, Any] | None:
        p = self._path(conv_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """Conversations (id, title, updated_at), most recent first."""
        if not self.dir.exists():
            return []
        items = []
        for p in self.dir.glob("*.json"):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items.append(
                {
                    "id": c["id"],
                    "title": c.get("title", ""),
                    "updated_at": c.get("updated_at", ""),
                    "n_messages": len(c.get("messages", [])),
                }
            )
        return sorted(items, key=lambda c: c["updated_at"], reverse=True)

    def delete(self, conv_id: str) -> None:
        p = self._path(conv_id)
        if p.exists():
            p.unlink()


def conversation_markdown(conv: dict[str, Any]) -> str:
    """Export of a conversation (questions and answers with citations)."""
    lines = [f"# {conv['title']}", f"*{conv.get('updated_at', '')[:16].replace('T', ' ')}*", ""]
    for m in conv["messages"]:
        who = "**Question**" if m["role"] == "user" else "**Réponse**"
        lines += [who, "", m.get("answer") or m["content"], ""]
    return "\n".join(lines)
