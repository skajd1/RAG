import asyncio
import datetime
import json
import os
import threading
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import requests

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "backend" else APP_DIR
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=True)

from database import VectorDB
from ingestion import ConfluenceIngestor, JiraIngestor
from rag_chain import RAGChain
from security import client_ip_from_headers, configured_allowed_networks, is_ip_allowed

app = FastAPI(title="MetsaBrain Intelligence Hub API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

allowed_internal_networks = configured_allowed_networks()


@app.middleware("http")
async def restrict_to_internal_network(request, call_next):
    fallback_host = request.client.host if request.client else None
    client_ip = client_ip_from_headers(request.headers, fallback_host)
    if not is_ip_allowed(client_ip, allowed_internal_networks):
        print(f"[SECURITY][ip-denied] client_ip={client_ip}")
        return JSONResponse(status_code=403, content={"detail": "Forbidden from this network."})
    return await call_next(request)


vector_db = VectorDB()
vector_db.init_collection(vector_size=4096)
rag_engine = RAGChain()
confluence_ingestor = ConfluenceIngestor()
jira_ingestor = JiraIngestor()

ingest_status = {"status": "idle", "last_space": None, "source_type": None, "processed_chunks": 0, "error": None}
ingest_history = []
scheduled_ingest_history = []
scheduled_ingest_last_runs = set()
scheduler_status = {
    "status": "idle",
    "current": None,
    "total": 0,
    "completed": 0,
    "failed": 0,
    "results": [],
    "started_at": None,
    "completed_at": None,
    "last_slot": None,
    "last_checked_at": None,
    "last_due_slot": None,
    "last_error": None,
}
scheduler_task = None
ingest_job_lock = threading.Lock()


class MentionContext(BaseModel):
    mention_type: str
    source_type: str
    page_id: str | None = None
    space: str | None = None
    title: str | None = None
    content_type: str | None = None


class ChatHistoryTurn(BaseModel):
    role: str
    content: str


class ChatSourceReference(BaseModel):
    title: str | None = None
    url: str | None = None
    page_id: str | None = None
    space: str | None = None
    space_name: str | None = None
    source_type: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    message: str
    mentions: list[MentionContext] = []
    history: list[ChatHistoryTurn] = []
    prior_sources: list[ChatSourceReference] = []


class IngestRequest(BaseModel):
    space_key: str
    source_type: str = "confluence"
    replace_existing: bool = True


@app.get("/")
async def root():
    return {"message": "MetsaBrain API is running"}


def is_scheduler_enabled():
    return os.getenv("INGEST_SCHEDULER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
async def start_scheduled_ingest_loop():
    global scheduler_task
    load_scheduled_ingest_state()
    if not is_scheduler_enabled():
        scheduler_status["status"] = "disabled"
        print("[INGEST][scheduler][startup] disabled by INGEST_SCHEDULER_ENABLED=false")
        return
    if scheduler_task and not scheduler_task.done():
        print("[INGEST][scheduler][startup] existing scheduler task is already running")
        return
    scheduler_task = asyncio.create_task(scheduled_ingest_loop())


@app.on_event("shutdown")
async def stop_scheduled_ingest_loop():
    if scheduler_task:
        scheduler_task.cancel()


def run_ingestion(space_key: str, source_type: str, replace_existing: bool = True):
    global ingest_status, ingest_history
    try:
        ingest_status["status"] = "processing"
        ingest_status["last_space"] = space_key
        ingest_status["source_type"] = source_type
        ingest_status["processed_chunks"] = 0
        ingest_status["error"] = None
        log_ingest("manual-start", source_type, space_key, replace_existing=replace_existing)
        selected_ingestor = jira_ingestor if source_type == "jira" else confluence_ingestor
        count = selected_ingestor.process_and_save(space_key, replace_existing=replace_existing)
        source_name = getattr(selected_ingestor, "last_source_name", None)
        log_ingest("manual-complete", source_type, space_key, source_name, chunks=count)
        ingest_status["processed_chunks"] = count
        ingest_status["status"] = "completed"
        ingest_status["error"] = None
        ingest_history.append(
            {
                "space": space_key,
                "space_name": source_name,
                "source_type": source_type,
                "chunks": count,
                "time": now_text(),
                "status": "success",
            }
        )
    except Exception as e:
        log_ingest("manual-failed", source_type, space_key, error=e)
        ingest_status["status"] = "failed"
        ingest_status["error"] = str(e)
        ingest_history.append(
            {
                "space": space_key,
                "source_type": source_type,
                "chunks": 0,
                "time": now_text(),
                "status": "failed",
                "error": str(e),
            }
        )


def run_reserved_ingestion(space_key: str, source_type: str, replace_existing: bool = True):
    try:
        run_ingestion(space_key, source_type, replace_existing)
    finally:
        ingest_job_lock.release()


def validate_source_type(source_type: str):
    normalized = source_type.lower()
    if normalized not in {"confluence", "jira"}:
        raise HTTPException(status_code=400, detail="source_type must be confluence or jira.")
    return normalized


def source_label(source_type: str, source_key: str, source_name: str | None = None):
    name_part = f"[{source_name}]" if source_name and source_name != source_key else ""
    return f"[{source_type}:{source_key}]{name_part}"


def log_ingest(stage: str, source_type: str, source_key: str, source_name: str | None = None, **fields):
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" {details}" if details else ""
    print(f"[INGEST]{source_label(source_type, source_key, source_name)}[{stage}]{suffix}")


def now_text():
    return schedule_now().strftime("%Y-%m-%d %H:%M:%S")


def get_schedule_timezone():
    timezone_name = os.getenv("INGEST_SCHEDULE_TIMEZONE", "Asia/Seoul")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        print(f"Invalid schedule timezone ignored: {timezone_name}. Falling back to Asia/Seoul.")
        return ZoneInfo("Asia/Seoul")


def schedule_now():
    return datetime.datetime.now(get_schedule_timezone())


def get_schedule_times():
    raw_times = os.getenv("INGEST_SCHEDULE_TIMES", "03:00,12:30")
    schedule_times = []
    for raw_time in raw_times.split(","):
        value = raw_time.strip()
        if not value:
            continue
        try:
            datetime.datetime.strptime(value, "%H:%M")
        except ValueError:
            print(f"Invalid schedule time ignored: {value}. Expected HH:MM.")
            continue
        schedule_times.append(value)
    return schedule_times or ["03:00", "12:30"]


def get_schedule_state_file():
    configured_path = os.getenv("INGEST_SCHEDULE_STATE_FILE")
    if configured_path:
        configured = Path(configured_path)
        return configured if configured.is_absolute() else PROJECT_ROOT / configured
    return PROJECT_ROOT / ".ingest_schedule_state.json"


def load_scheduled_ingest_state():
    state_file = get_schedule_state_file()
    if not state_file.exists():
        return
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        for slot in data.get("last_runs", []):
            scheduled_ingest_last_runs.add(slot)
    except Exception as e:
        print(f"[INGEST][scheduler][state-load-failed] path={state_file} error={e}")


def save_scheduled_ingest_state():
    state_file = get_schedule_state_file()
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"last_runs": sorted(scheduled_ingest_last_runs)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[INGEST][scheduler][state-save-failed] path={state_file} error={e}")


def find_due_schedule_slots(now: datetime.datetime):
    catchup_enabled = os.getenv("INGEST_SCHEDULE_CATCHUP", "true").lower() in {"1", "true", "yes", "on"}
    window_minutes = int(os.getenv("INGEST_SCHEDULE_WINDOW_MINUTES", "10"))
    due_slots = []

    for schedule_time in get_schedule_times():
        hour, minute = [int(part) for part in schedule_time.split(":")]
        scheduled_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        elapsed_seconds = (now - scheduled_at).total_seconds()
        slot_key = f"{now.strftime('%Y-%m-%d')} {schedule_time}"
        if slot_key in scheduled_ingest_last_runs:
            continue
        if catchup_enabled and elapsed_seconds >= 0:
            due_slots.append((scheduled_at, slot_key))
        elif 0 <= elapsed_seconds < window_minutes * 60:
            due_slots.append((scheduled_at, slot_key))
    return [slot_key for _, slot_key in sorted(due_slots, key=lambda item: item[0])]


def find_due_schedule_slot(now: datetime.datetime):
    due_slots = find_due_schedule_slots(now)
    return due_slots[-1] if due_slots else None


def get_scheduled_targets():
    restored_history = recover_ingest_history_from_qdrant()
    targets = []
    seen = set()
    for item in restored_history:
        space = item.get("space")
        source_type = item.get("source_type", "confluence")
        if not space or source_type not in {"confluence", "jira"}:
            continue
        key = f"{source_type}:{space}"
        if key in seen:
            continue
        seen.add(key)
        targets.append({"space": space, "space_name": item.get("space_name"), "source_type": source_type})
    return targets


def run_scheduled_ingestion(slot_key: str | None = None):
    global scheduler_status, ingest_status

    targets = get_scheduled_targets()
    started_at = now_text()
    print(f"[INGEST][schedule:{slot_key or 'unknown'}][start] targets={len(targets)}")
    previous_last_checked_at = scheduler_status.get("last_checked_at")
    scheduler_status = {
        "status": "processing",
        "current": None,
        "total": len(targets),
        "completed": 0,
        "failed": 0,
        "results": [],
        "started_at": started_at,
        "completed_at": None,
        "last_slot": slot_key,
        "last_checked_at": previous_last_checked_at,
        "last_due_slot": slot_key,
        "last_error": None,
    }

    for target in targets:
        current = {"space": target["space"], "space_name": target.get("space_name"), "source_type": target["source_type"]}
        scheduler_status["current"] = current
        ingest_status["status"] = "processing"
        ingest_status["last_space"] = target["space"]
        ingest_status["source_type"] = target["source_type"]
        ingest_status["processed_chunks"] = 0
        log_ingest("schedule-source-start", target["source_type"], target["space"], target.get("space_name"), slot=slot_key)

        try:
            selected_ingestor = jira_ingestor if target["source_type"] == "jira" else confluence_ingestor
            count = selected_ingestor.process_and_save(target["space"], replace_existing=True)
            source_name = getattr(selected_ingestor, "last_source_name", None) or target.get("space_name")
            log_ingest("schedule-source-complete", target["source_type"], target["space"], source_name, slot=slot_key, chunks=count)
            result = {
                **current,
                "space_name": source_name,
                "chunks": count,
                "time": now_text(),
                "status": "success",
                "schedule_slot": slot_key,
            }
            ingest_status["processed_chunks"] = count
            ingest_status["status"] = "completed"
            scheduler_status["completed"] += 1
        except Exception as e:
            log_ingest("schedule-source-failed", target["source_type"], target["space"], target.get("space_name"), slot=slot_key, error=e)
            result = {
                **current,
                "space_name": target.get("space_name"),
                "chunks": 0,
                "time": now_text(),
                "status": "failed",
                "error": str(e),
                "schedule_slot": slot_key,
            }
            ingest_status["status"] = f"error: {str(e)}"
            scheduler_status["failed"] += 1

        scheduler_status["results"].append(result)
        ingest_history.append(result)
        scheduled_ingest_history.append(result)

    if slot_key:
        scheduled_ingest_last_runs.add(slot_key)
        save_scheduled_ingest_state()

    scheduler_status = {
        **scheduler_status,
        "status": "idle" if scheduler_status["failed"] == 0 else "idle_with_errors",
        "current": None,
        "completed_at": now_text(),
    }

    if not targets:
        print(f"[INGEST][schedule:{slot_key or 'unknown'}][skipped] reason=no-indexed-sources")
        scheduled_ingest_history.append(
            {
                "space": None,
                "source_type": None,
                "chunks": 0,
                "time": now_text(),
                "status": "skipped",
                "message": "No indexed Confluence spaces or Jira projects found.",
                "schedule_slot": slot_key,
            }
        )
        scheduler_status["status"] = "idle"

    print(
        f"[INGEST][schedule:{slot_key or 'unknown'}][complete] "
        f"completed={scheduler_status['completed']} failed={scheduler_status['failed']} total={scheduler_status['total']}"
    )


def run_reserved_scheduled_ingestion(slot_key: str | None = None):
    try:
        run_scheduled_ingestion(slot_key)
    finally:
        ingest_job_lock.release()


def get_scheduler_overview():
    current_schedule_now = schedule_now()
    return {
        "scheduler": scheduler_status,
        "schedule_times": get_schedule_times(),
        "schedule_timezone": os.getenv("INGEST_SCHEDULE_TIMEZONE", "Asia/Seoul"),
        "schedule_now": current_schedule_now.strftime("%Y-%m-%d %H:%M:%S"),
        "due_slot": find_due_schedule_slot(current_schedule_now),
        "scheduler_enabled": is_scheduler_enabled(),
        "schedule_catchup": os.getenv("INGEST_SCHEDULE_CATCHUP", "true").lower() in {"1", "true", "yes", "on"},
        "schedule_window_minutes": int(os.getenv("INGEST_SCHEDULE_WINDOW_MINUTES", "10")),
        "tracked_sources": get_scheduled_targets(),
        "last_runs": sorted(scheduled_ingest_last_runs),
    }


async def scheduled_ingest_loop():
    poll_seconds = int(os.getenv("INGEST_SCHEDULER_POLL_SECONDS", "30"))
    while True:
        try:
            scheduler_status["last_checked_at"] = now_text()
            scheduler_status["last_error"] = None

            due_slot = find_due_schedule_slot(schedule_now())
            due_slots = find_due_schedule_slots(schedule_now())
            scheduler_status["last_due_slot"] = due_slot
            if due_slot:
                if not ingest_job_lock.acquire(blocking=False):
                    await asyncio.sleep(poll_seconds)
                    continue
                skipped_slots = due_slots[:-1]
                if skipped_slots:
                    scheduled_ingest_last_runs.update(skipped_slots)
                    save_scheduled_ingest_state()
                    print(f"[INGEST][scheduler][catchup-skip-old-slots] slots={skipped_slots} selected={due_slot}")
                await asyncio.to_thread(run_reserved_scheduled_ingestion, due_slot)
                continue
        except Exception as e:
            scheduler_status["last_error"] = str(e)
            print(f"[SCHEDULER] loop error: {e}")

        await asyncio.sleep(poll_seconds)


def recover_ingest_history_from_qdrant():
    space_stats = {}
    offset = None

    while True:
        points, offset = vector_db.client.scroll(
            collection_name=vector_db.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            space = metadata.get("space")
            if not space:
                continue
            space_name = metadata.get("space_name")
            source_type = metadata.get("source_type", "confluence")

            stat_key = f"{source_type}:{space}"
            stat = space_stats.setdefault(
                stat_key,
                {
                    "space": space,
                    "space_name": space_name,
                    "source_type": source_type,
                    "chunks": 0,
                    "time": "restored from Qdrant",
                    "status": "indexed",
                },
            )
            stat["chunks"] += 1
            if not stat.get("space_name") and space_name:
                stat["space_name"] = space_name

        if offset is None:
            break

    return sorted(space_stats.values(), key=lambda item: item["space"])


def merge_ingest_history(memory_history, restored_history):
    merged = {f"{item.get('source_type', 'confluence')}:{item['space']}": item for item in restored_history}
    for item in memory_history:
        key = f"{item.get('source_type', 'confluence')}:{item['space']}"
        if not item.get("space_name") and key in merged:
            item = {**item, "space_name": merged[key].get("space_name")}
        merged[key] = item
    return list(merged.values())


def normalize_search_text(value: str | None):
    import re

    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", value or "").lower()


def mention_score(query_text: str, metadata: dict, page_content: str):
    space = metadata.get("space") or ""
    space_name = metadata.get("space_name") or ""
    breadcrumb = metadata.get("breadcrumb") or ""
    normalized_query = normalize_search_text(query_text)
    if not normalized_query:
        return 0

    space_text = normalize_search_text(space)
    space_name_text = normalize_search_text(space_name)
    breadcrumb_text = normalize_search_text(breadcrumb)
    content_text = normalize_search_text(page_content[:2000])

    score = 0
    if normalized_query == space_text or normalized_query == space_name_text:
        score += 80
    elif normalized_query in space_text or normalized_query in space_name_text:
        score += 60
    if normalized_query in breadcrumb_text:
        score += 25
    if normalized_query in content_text:
        score += 10
    return score


def find_mentions(query: str, limit: int):
    spaces = {}
    offset = None

    while True:
        points, offset = vector_db.client.scroll(
            collection_name=vector_db.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            page_content = payload.get("page_content") or ""
            score = mention_score(query, metadata, page_content)
            if score <= 0:
                continue

            source_type = metadata.get("source_type", "confluence")
            space = metadata.get("space")
            space_name = metadata.get("space_name") or space
            if not space:
                continue

            space_key = f"{source_type}:{space}"
            current = spaces.get(space_key)
            source_label = "Jira 프로젝트" if source_type == "jira" else "Confluence 스페이스"
            if not current or score > current["score"]:
                spaces[space_key] = {
                    "mention_type": "space",
                    "source_type": source_type,
                    "space": space,
                    "space_name": space_name,
                    "title": space_name,
                    "subtitle": f"{source_label} · {space}",
                    "content_type": "space",
                    "score": score,
                }

        if offset is None:
            break

    results = list(spaces.values())
    results.sort(key=lambda item: item["score"], reverse=True)
    return [{key: value for key, value in item.items() if key != "score"} for item in results[: max(min(limit, 20), 1)]]


@app.get("/search/mentions")
async def search_mentions(q: str, limit: int = 8):
    query = q.strip()
    if not query:
        return []
    return await asyncio.to_thread(find_mentions, query, limit)


@app.get("/ingest/history")
async def get_ingest_history():
    try:
        restored_history = await asyncio.to_thread(recover_ingest_history_from_qdrant)
        return merge_ingest_history(ingest_history, restored_history)
    except Exception as e:
        print(f"Failed to recover ingest history from Qdrant: {e}")
        return ingest_history


@app.get("/ingest/spaces")
async def get_indexed_spaces():
    try:
        return await asyncio.to_thread(recover_ingest_history_from_qdrant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/sources/search")
async def search_ingest_sources(q: str, limit: int = 10, source_type: str | None = None):
    query = q.strip()
    if not query:
        return []

    result_limit = max(1, min(limit, 20))
    if source_type:
        normalized_source_type = validate_source_type(source_type)
        try:
            if normalized_source_type == "jira":
                return await asyncio.to_thread(jira_ingestor.search_projects, query, result_limit)
            return await asyncio.to_thread(confluence_ingestor.search_spaces, query, result_limit)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    searches = await asyncio.gather(
        asyncio.to_thread(confluence_ingestor.search_spaces, query, result_limit),
        asyncio.to_thread(jira_ingestor.search_projects, query, result_limit),
        return_exceptions=True,
    )
    successful_results = [results for results in searches if not isinstance(results, Exception)]
    if not successful_results:
        errors = "; ".join(str(error) for error in searches if isinstance(error, Exception))
        raise HTTPException(status_code=502, detail=errors)

    merged = []
    for index in range(result_limit):
        for results in successful_results:
            if index < len(results):
                merged.append(results[index])
                if len(merged) >= result_limit:
                    return merged
    return merged


@app.delete("/ingest/clear")
async def clear_all_data():
    if not ingest_job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion job is currently in progress.")
    try:
        await asyncio.to_thread(vector_db.client.delete_collection, vector_db.collection_name)
        await asyncio.to_thread(vector_db.init_collection, vector_size=4096)
        global ingest_history
        ingest_history = []
        return {"status": "success", "message": "All vector data cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ingest_job_lock.release()


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    source_type = validate_source_type(request.source_type)
    if not ingest_job_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Another ingestion job is already in progress."}
    ingest_status.update(
        {
            "status": "processing",
            "last_space": request.space_key,
            "source_type": source_type,
            "processed_chunks": 0,
            "error": None,
        }
    )
    try:
        background_tasks.add_task(run_reserved_ingestion, request.space_key, source_type, request.replace_existing)
    except Exception:
        ingest_job_lock.release()
        raise
    return {"status": "started", "message": f"Started {source_type} ingestion for '{request.space_key}'"}


@app.get("/ingest/status")
async def get_ingest_status():
    return ingest_status


@app.get("/ingest/schedules")
async def get_ingest_schedules():
    return await asyncio.to_thread(get_scheduler_overview)


@app.post("/ingest/schedules/run")
async def run_ingest_schedule_now(background_tasks: BackgroundTasks):
    if not ingest_job_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Another ingestion job is already in progress."}
    scheduler_status["status"] = "processing"
    scheduler_status["current"] = None
    scheduler_status["last_error"] = None
    try:
        background_tasks.add_task(run_reserved_scheduled_ingestion, "manual")
    except Exception:
        ingest_job_lock.release()
        raise
    return {"status": "started", "message": "Started scheduled ingestion for all indexed sources."}


@app.get("/ingest/schedules/history")
async def get_scheduled_ingest_history():
    return scheduled_ingest_history


@app.post("/chat")
@app.post("/chat/")
async def chat(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System is still initializing.")
    try:
        print(f"Received chat request: chars={len(request.message)} mentions={len(request.mentions)}")
        return StreamingResponse(
            rag_engine.stream_ask(
                request.message,
                [mention.dict() for mention in request.mentions],
                [turn.dict() for turn in request.history],
                [source.dict() for source in request.prior_sources],
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat")
async def chat_get_info():
    return {"message": "Chat endpoint only supports POST requests for queries."}


@app.get("/health")
async def health_check():
    health = {
        "status": "healthy",
        "branding": "MetsaBrain",
        "components": {
            "api": "ok",
            "vector_db": "unknown",
            "llm_engine": "unknown",
        },
    }
    try:
        await asyncio.to_thread(vector_db.client.get_collections)
        health["components"]["vector_db"] = "ok"
    except Exception:
        health["components"]["vector_db"] = "error"

    try:
        ollama_host = getattr(rag_engine, "ollama_host", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        ollama_timeout = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "3"))
        response = await asyncio.to_thread(requests.get, f"{ollama_host}/api/tags", timeout=ollama_timeout)
        response.raise_for_status()
        health["components"]["llm_engine"] = "ok"
    except Exception:
        health["components"]["llm_engine"] = "error"

    if any(component == "error" for component in health["components"].values()):
        health["status"] = "degraded"

    return health
