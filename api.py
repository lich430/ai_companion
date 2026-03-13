from __future__ import annotations

import logging
import os
from queue import Empty, Full, Queue
from threading import Thread, Lock
from typing import List, Optional
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import CompanionEngine
from engine.cadence_simulator import CadenceSimulator


logger = logging.getLogger("uvicorn.error")
INCOMING_BATCH_WAIT_SEC = float(os.getenv("INCOMING_BATCH_WAIT_SEC", "40"))
BLOCK_ALL_GROUP_MESSAGES = os.getenv("BLOCK_ALL_GROUP_MESSAGES", "false").lower() == "true"
BLOCK_GROUP_IDS = {x.strip() for x in os.getenv("BLOCK_GROUP_IDS", "").split(",") if x.strip()}
BLOCK_GROUP_KEYWORDS = {x.strip() for x in os.getenv("BLOCK_GROUP_KEYWORDS", "").split(",") if x.strip()}
INCOMING_DEDUP_WINDOW_SEC = max(0.0, float(os.getenv("INCOMING_DEDUP_WINDOW_SEC", "3")))
_incoming_dedup_seen: dict[tuple[str, str], float] = {}
_incoming_dedup_lock = Lock()


def _is_group_chat(value: Optional[str]) -> bool:
    raw = str(value or "").strip()
    return bool(raw) and raw.endswith("@chatroom")


def _should_filter_group_message(username: str, user_id: Optional[str]) -> bool:
    uid = str(user_id or username or "").strip()
    name = str(username or "").strip()
    if not (_is_group_chat(uid) or _is_group_chat(name)):
        return False
    if BLOCK_ALL_GROUP_MESSAGES:
        return True
    if uid in BLOCK_GROUP_IDS or name in BLOCK_GROUP_IDS:
        return True
    return any((kw in uid) or (kw in name) for kw in BLOCK_GROUP_KEYWORDS)


def _is_duplicate_incoming(username: str, user_id: Optional[str], message: str) -> bool:
    if INCOMING_DEDUP_WINDOW_SEC <= 0:
        return False
    uid = str(user_id or username or "").strip()
    msg = str(message or "").strip()
    if not uid or not msg:
        return False
    key = (uid, msg)
    now = time.time()
    cutoff = now - INCOMING_DEDUP_WINDOW_SEC
    with _incoming_dedup_lock:
        if len(_incoming_dedup_seen) > 10000:
            stale = [k for k, ts in _incoming_dedup_seen.items() if ts < cutoff]
            for k in stale:
                _incoming_dedup_seen.pop(k, None)
        last = _incoming_dedup_seen.get(key)
        _incoming_dedup_seen[key] = now
    return last is not None and (now - last) <= INCOMING_DEDUP_WINDOW_SEC


class ChatRequest(BaseModel):
    user_id: str
    message: str
    with_cadence: bool = Field(default=True, description="鏄惁杩斿洖鍒嗘/寤惰繜璁″垝")


class CadencePart(BaseModel):
    text: str
    delay_ms: int


class ChatResponse(BaseModel):
    type: str = "reply"
    reply: str
    recommended_delay_ms: int = 0
    reason: str = ""
    state: dict
    cadence: List[CadencePart] = []


class SyncMessagesRequest(BaseModel):
    user_id: str
    messages: List[str]


class SyncMessagesResponse(BaseModel):
    user_id: str
    synced_count: int
    state: dict


class ProactiveRequest(BaseModel):
    user_id: str
    force: bool = False
    with_cadence: bool = True


class ProactiveResponse(BaseModel):
    should_send: bool
    reason: str
    text: str = ""
    cadence: List[CadencePart] = []
    state: dict


class ProactiveMarkRequest(BaseModel):
    user_id: str


class ContextMessage(BaseModel):
    role: str
    content: str
    ts: str


class ContextResponse(BaseModel):
    user_id: str
    state: dict
    message_count: int
    messages: List[ContextMessage]


class UserItem(BaseModel):
    user_id: str
    turn_count: int
    message_count: int
    last_message_ts: Optional[str] = None


class UsersResponse(BaseModel):
    total: int
    users: List[UserItem]


class HookIncomingRequest(BaseModel):
    username: str
    message: str
    user_id: Optional[str] = None


class HookIncomingResponse(BaseModel):
    ok: bool
    reason: str = ""
    task_id: Optional[int] = None
    queue_size: int = 0
    queued: Optional[dict] = None
    state: dict = {}


class HookPendingItem(BaseModel):
    id: int
    user_id: str
    username: str
    reply: str
    recommended_delay_ms: int = 0
    created_at: str


class HookPendingResponse(BaseModel):
    total: int
    items: List[HookPendingItem]


class ReplyPolicyStatsResponse(BaseModel):
    stats: dict


app = FastAPI(title="Guoguo AI Companion API", version="1.2.0")
engine = CompanionEngine()
cadence_sim = CadenceSimulator()
incoming_queue: Queue = Queue(maxsize=5000)
_worker_started = False
_task_seq = 0
_task_lock = Lock()


def _next_task_id() -> int:
    global _task_seq
    with _task_lock:
        _task_seq += 1
        return _task_seq


def _incoming_worker():
    logger.info("incoming worker started")
    while True:
        try:
            first = incoming_queue.get(timeout=1.0)
        except Empty:
            continue
        consumed = [first]
        same_user_tasks = [first]
        buffered = []
        try:
            base_uid = str(first.get("user_id") or first.get("username") or "")
            # Debounce window: wait a bit and merge same-user messages in one GLM call.
            deadline = time.time() + max(0.0, INCOMING_BATCH_WAIT_SEC)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    nxt = incoming_queue.get(timeout=min(1.0, remaining))
                except Empty:
                    continue
                consumed.append(nxt)
                uid = str(nxt.get("user_id") or nxt.get("username") or "")
                if uid == base_uid:
                    same_user_tasks.append(nxt)
                else:
                    buffered.append(nxt)

            # Put back other users' tasks, keep overall queue behavior.
            for item in buffered:
                incoming_queue.put_nowait(item)

            messages = [str(x.get("message", "")).strip() for x in same_user_tasks if str(x.get("message", "")).strip()]
            usernames = [str(x.get("username", "")).strip() for x in same_user_tasks if str(x.get("username", "")).strip()]
            username = usernames[0] if usernames else base_uid
            task_ids = [x.get("task_id") for x in same_user_tasks]
            logger.info(
                f"incoming worker processing batch: task_ids={task_ids}, "
                f"username={username!r}, user_id={base_uid!r}, "
                f"messages_count={len(messages)}, wait_sec={INCOMING_BATCH_WAIT_SEC}"
            )
            result = engine.ingest_incoming_messages(
                username=username,
                messages=messages,
                user_id=(base_uid if base_uid else None),
            )
            queued_items = result.get("queued_items") or []
            logger.info(
                "incoming worker done batch: task_ids=%s, reason=%s, queued_items=%d",
                task_ids,
                result.get("reason"),
                len(queued_items),
            )
            if queued_items:
                replies = [str(item.get("reply", "")).strip() for item in queued_items]
                delays = [int(item.get("recommended_delay_ms", 0) or 0) for item in queued_items]
                logger.info(
                    "incoming worker model replies: task_ids=%s, replies=%s, recommended_delay_ms=%s",
                    task_ids,
                    replies,
                    delays,
                )
            else:
                logger.info(
                    "incoming worker no model reply: task_ids=%s, reason=%s",
                    task_ids,
                    result.get("reason"),
                )
        except Exception as e:
            logger.error(f"incoming worker failed batch: task_ids={[x.get('task_id') for x in consumed]} err={e}")
        finally:
            # mark all consumed tasks done (buffered ones were re-enqueued)
            for _ in consumed:
                incoming_queue.task_done()


@app.on_event("startup")
def start_background_workers():
    global _worker_started
    if _worker_started:
        return
    Thread(target=_incoming_worker, daemon=True).start()
    _worker_started = True


@app.get("/health")
def health():
    return {
        "ok": True,
        "role_name": engine.role_name,
        "role_id": engine.role_id,
        "role_bible_path": engine.role_bible_path,
        "llm_provider": getattr(engine, "llm_provider", "glm"),
        "chat_model": engine.generator.model,
        "glm_chat_model": engine.generator.model,
        "incoming_batch_wait_sec": INCOMING_BATCH_WAIT_SEC,
        "block_all_group_messages": BLOCK_ALL_GROUP_MESSAGES,
        "block_group_ids_count": len(BLOCK_GROUP_IDS),
        "block_group_keywords_count": len(BLOCK_GROUP_KEYWORDS),
        "incoming_dedup_window_sec": INCOMING_DEDUP_WINDOW_SEC,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(
        f"/chat request: user_id={req.user_id}, with_cadence={req.with_cadence}, message={req.message!r}"
    )
    result = engine.chat(req.user_id, req.message)
    state = engine.get_state_snapshot(req.user_id)
    reply = str(result.get("text", "") or "")

    cadence = []
    if req.with_cadence and result.get("type") == "reply" and reply:
        plan = cadence_sim.build_plan(
            reply=reply,
            mood=str(state.get("mood", "neutral")),
            stage=str(state.get("stage", "stranger")),
        )
        cadence = [CadencePart(**x) for x in plan]

    return ChatResponse(
        type=str(result.get("type", "reply")),
        reply=reply,
        recommended_delay_ms=int(result.get("recommended_delay_ms", 0) or 0),
        reason=str(result.get("reason", "")),
        state=state,
        cadence=cadence,
    )


@app.post("/chat/sync_user_messages", response_model=SyncMessagesResponse)
def sync_user_messages(req: SyncMessagesRequest):
    logger.info(
        f"/chat/sync_user_messages request: user_id={req.user_id}, messages_count={len(req.messages)}"
    )
    data = engine.sync_user_messages(req.user_id, req.messages)
    return SyncMessagesResponse(**data)


@app.post("/proactive/suggest", response_model=ProactiveResponse)
def proactive_suggest(req: ProactiveRequest):
    logger.info(
        f"/proactive/suggest request: user_id={req.user_id}, force={req.force}, with_cadence={req.with_cadence}"
    )
    plan = engine.suggest_proactive_message(req.user_id, force=req.force)
    state = engine.get_state_snapshot(req.user_id)

    cadence = []
    if req.with_cadence and plan.should_send and plan.text:
        split = cadence_sim.build_plan(
            reply=plan.text,
            mood=str(state.get("mood", "neutral")),
            stage=str(state.get("stage", "stranger")),
        )
        cadence = [CadencePart(**x) for x in split]

    return ProactiveResponse(
        should_send=plan.should_send,
        reason=plan.reason,
        text=plan.text,
        cadence=cadence,
        state=state,
    )


@app.post("/proactive/mark_sent")
def proactive_mark_sent(req: ProactiveMarkRequest):
    logger.info(f"/proactive/mark_sent request: user_id={req.user_id}")
    engine.mark_proactive_sent(req.user_id)
    return {"ok": True}


@app.get("/context/{user_id}", response_model=ContextResponse)
def get_context(user_id: str, limit: int = 20):
    logger.info(f"/context request: user_id={user_id}, limit={limit}")
    data = engine.get_chat_context(user_id=user_id, limit=limit)
    return ContextResponse(**data)


@app.get("/users", response_model=UsersResponse)
def get_users():
    logger.info("/users request")
    users = engine.list_users()
    return UsersResponse(total=len(users), users=users)


@app.post("/hook/incoming", response_model=HookIncomingResponse)
def hook_incoming(req: HookIncomingRequest):
    logger.info(
        f"/hook/incoming request: username={req.username!r}, user_id={req.user_id!r}, message={req.message!r}"
    )
    if _should_filter_group_message(req.username, req.user_id):
        logger.info(
            "/hook/incoming filtered group message: username=%r, user_id=%r",
            req.username,
            req.user_id,
        )
        return HookIncomingResponse(
            ok=True,
            reason="filtered_group",
            queue_size=incoming_queue.qsize(),
        )
    if _is_duplicate_incoming(req.username, req.user_id, req.message):
        logger.info(
            "/hook/incoming duplicate dropped: username=%r, user_id=%r, message=%r",
            req.username,
            req.user_id,
            req.message,
        )
        return HookIncomingResponse(
            ok=True,
            reason="duplicate_message",
            queue_size=incoming_queue.qsize(),
        )
    task_id = _next_task_id()
    task = {
        "task_id": task_id,
        "username": req.username,
        "user_id": req.user_id,
        "message": req.message,
    }
    try:
        incoming_queue.put_nowait(task)
    except Full:
        return HookIncomingResponse(
            ok=False,
            reason="queue_full",
            task_id=task_id,
            queue_size=incoming_queue.qsize(),
        )
    return HookIncomingResponse(
        ok=True,
        reason="queued",
        task_id=task_id,
        queue_size=incoming_queue.qsize(),
        queued={"username": req.username, "user_id": req.user_id, "message": req.message},
    )


@app.get("/hook/pending", response_model=HookPendingResponse)
def hook_pending(limit: int = 1, pop: bool = True):
    logger.info(f"/hook/pending request: limit={limit}, pop={pop}")
    items = engine.fetch_pending_replies(limit=limit, pop=pop)
    return HookPendingResponse(total=len(items), items=items)


@app.get("/metrics/reply_policy", response_model=ReplyPolicyStatsResponse)
def get_reply_policy_stats():
    logger.info("/metrics/reply_policy request")
    return ReplyPolicyStatsResponse(stats=engine.get_reply_policy_stats())

