"""本地网页任务服务：持久队列、单 worker、独立数据库连接。"""
import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from research.runtime import TaskBusy, TaskFileLock

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WebSettings:
    db_path: Path = ROOT / "data" / "memory.db"
    index_path: Path = ROOT / "data" / "qdrant"
    model_cache: Path = ROOT / "data" / "models"
    context_limit: int = 16000

    def __post_init__(self):
        for name in ("db_path", "index_path", "model_cache"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())


class WebError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class WorkerStopped(RuntimeError):
    pass


@contextmanager
def open_runtime(settings):
    # 延迟导入，模块加载阶段不初始化模型或访问数据库。
    from langgraph.checkpoint.sqlite import SqliteSaver
    from research.engine import TeamController, TeamWorkflowAgent
    from research.generation import GenerationStore
    from research.management import BudgetLedger
    with ExitStack() as stack:
        conn = stack.enter_context(closing(sqlite3.connect(
            settings.db_path, timeout=10, check_same_thread=False
        )))
        checkpoint = stack.enter_context(closing(sqlite3.connect(
            settings.db_path.with_name("research_checkpoints.db"), timeout=10,
            check_same_thread=False,
        )))
        store, ledger = GenerationStore(conn), BudgetLedger(conn)
        agent = TeamWorkflowAgent(
            store, SqliteSaver(checkpoint), ledger, db_path=settings.db_path,
            index_path=settings.index_path, model_cache=settings.model_cache,
            context_limit=settings.context_limit,
        )
        controller = TeamController(store, agent, {agent.VERSION: agent}, settings.db_path)
        yield SimpleNamespace(store=store, agent=agent, controller=controller, ledger=ledger)


def now():
    return datetime.now(timezone.utc).isoformat()


class TaskService:
    def __init__(self, settings=None, runtime_factory=None):
        self.settings = settings or WebSettings()
        self.runtime_factory = runtime_factory or open_runtime
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = self._lease = None
        self._worker_error = ""
        with self.runtime_factory(self.settings):
            pass
        with self._db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS web_jobs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES research_tasks(id),
                    action TEXT NOT NULL CHECK(action IN ('create','approval','resume')),
                    request_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued','running','done','failed','interrupted')),
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS web_one_active_job
                ON web_jobs(task_id) WHERE status IN ('queued','running');
            """)

    @contextmanager
    def _db(self, timeout=10):
        with closing(sqlite3.connect(self.settings.db_path, timeout=timeout)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn

    @staticmethod
    def _request(request_id, action, payload):
        if not isinstance(request_id, str) or not 1 <= len(request_id.strip()) <= 128:
            raise WebError(422, "request_id 必须是 1～128 字符的非空字符串")
        raw = json.dumps([action, payload], sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
        return request_id.strip(), hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _receipt(row):
        return {"task_id": row["task_id"], "job_id": row["id"]}

    def _repeat(self, conn, request_id, fingerprint):
        row = conn.execute("SELECT * FROM web_jobs WHERE request_id=?", (request_id,)).fetchone()
        if row is not None and row["fingerprint"] != fingerprint:
            raise WebError(409, "该请求编号已用于不同操作，请重新提交")
        return self._receipt(row) if row is not None else None

    @staticmethod
    def _insert_job(conn, task_id, action, request_id, fingerprint, payload):
        job_id, stamp = uuid4().hex, now()
        conn.execute(
            "INSERT INTO web_jobs VALUES (?,?,?,?,?,?,'queued','',?,?)",
            (job_id, task_id, action, request_id, fingerprint,
             json.dumps(payload, ensure_ascii=False), stamp, stamp),
        )
        return {"task_id": task_id, "job_id": job_id}

    def _check_task(self, conn, task_id):
        # 第一阶段仅管理网页创建的 default 研究任务。
        row = conn.execute(
            "SELECT 1 FROM research_task_owners o WHERE o.task_id=? AND o.user_id='default' "
            "AND EXISTS(SELECT 1 FROM web_jobs w WHERE w.task_id=o.task_id AND w.action='create')",
            (task_id,),
        ).fetchone()
        if row is None:
            raise WebError(404, "网页研究任务不存在")

    def create(self, question, request_id):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
            raise WebError(422, "研究问题必须为 1～1000 个字符")
        question = question.strip()
        request_id, fingerprint = self._request(request_id, "create", question)
        with self._db() as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            repeated = self._repeat(conn, request_id, fingerprint)
            if repeated is not None:
                return repeated
            task_id = uuid4().hex
            conn.execute(
                "INSERT INTO research_tasks(id,question,status,created_at) VALUES (?,?,'planning',?)",
                (task_id, question, now()),
            )
            conn.execute("INSERT INTO research_task_owners VALUES (?,'default')", (task_id,))
            result = self._insert_job(conn, task_id, "create", request_id, fingerprint, {})
        self._wake.set()
        return result

    @staticmethod
    def _job(row):
        return None if row is None else {key: row[key] for key in ("id", "action", "status", "error")}

    def _jobs(self, conn, task_id):
        active = conn.execute(
            "SELECT * FROM web_jobs WHERE task_id=? AND status IN ('queued','running')", (task_id,)
        ).fetchone()
        latest = conn.execute(
            "SELECT * FROM web_jobs WHERE task_id=? ORDER BY rowid DESC LIMIT 1", (task_id,)
        ).fetchone()
        return self._job(active), self._job(latest)

    @staticmethod
    def _pending(agent, task_id, info):
        if not info["pending"]:
            return None
        snapshot = agent._snapshot(task_id)
        token = snapshot.config["configurable"]["checkpoint_id"]
        return {**info["pending"][0], "token": token}

    def detail(self, task_id):
        with self.runtime_factory(self.settings) as rt:
            self._check_task(rt.store.conn, task_id)
            task = rt.store.get_task(task_id)
            with self._db() as conn:
                active, latest = self._jobs(conn, task_id)
            if active:
                # 后台可能正在提交 checkpoint，轮询只读业务快照。
                return {
                    **task, "active_job": active, "last_job": latest, "pending": None,
                    "resume_allowed": False, "usage": rt.ledger.get(task_id),
                }
            info = rt.controller.inspect(task_id)
            pending = self._pending(rt.agent, task_id, info)
            interrupted_plan = bool(
                latest and latest["status"] == "interrupted"
                and task["status"] == "planning" and not info["exists"]
            )
            return {
                **task, "active_job": active, "last_job": latest, "pending": pending,
                "resume_allowed": bool(not active and not pending and (
                    info["resume_allowed"] or interrupted_plan
                )), "usage": rt.ledger.get(task_id),
            }

    def list_tasks(self, limit=50, offset=0):
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
            raise WebError(422, "分页参数无效")
        with self._db() as conn:
            scope = "EXISTS(SELECT 1 FROM web_jobs w WHERE w.task_id=t.id AND w.action='create')"
            total = conn.execute("SELECT COUNT(*) FROM research_tasks t WHERE " + scope).fetchone()[0]
            rows = conn.execute(
                "SELECT t.id,t.question,t.status,t.created_at,"
                "(SELECT j.status FROM web_jobs j WHERE j.task_id=t.id ORDER BY j.rowid DESC LIMIT 1) "
                "AS job_status FROM research_tasks t WHERE " + scope +
                " ORDER BY t.created_at DESC,t.id DESC LIMIT ? OFFSET ?", (limit, offset),
            ).fetchall()
            return {"items": [dict(row) for row in rows], "total": total}

    def _enqueue(self, task_id, action, payload, request_id):
        request_id, fingerprint = self._request(request_id, action, [task_id, payload])
        # 已接受请求的重试，即使 worker 正持任务锁，也返回同一收据。
        with self._db() as conn:
            self._check_task(conn, task_id)
            repeated = self._repeat(conn, request_id, fingerprint)
            if repeated is not None:
                return repeated
        # 网页等待用户时不持锁；提交操作时短暂锁定，worker 执行时再次校验。
        try:
            with TaskFileLock(self.settings.db_path, task_id):
                with self._db() as conn:
                    self._check_task(conn, task_id)
                    repeated = self._repeat(conn, request_id, fingerprint)
                    if repeated is not None:
                        return repeated
                view = self.detail(task_id)
                if view["active_job"]:
                    raise WebError(409, "任务已经排队或正在执行")
                if action == "approval":
                    if not view["pending"] or view["pending"]["token"] != payload["token"]:
                        raise WebError(409, "审批已变化，请刷新后再操作")
                elif not view["resume_allowed"]:
                    raise WebError(409, "该任务没有可恢复的执行步骤")
                with self._db() as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    repeated = self._repeat(conn, request_id, fingerprint)
                    result = repeated or self._insert_job(
                        conn, task_id, action, request_id, fingerprint, payload
                    )
        except TaskBusy as error:
            raise WebError(409, "任务正在处理，请稍后刷新") from error
        except sqlite3.IntegrityError as error:
            raise WebError(409, "任务已收到其他操作，请刷新状态") from error
        self._wake.set()
        return result

    def approve(self, task_id, token, approved, request_id):
        if type(approved) is not bool or not isinstance(token, str) or not token:
            raise WebError(422, "审批参数无效")
        return self._enqueue(task_id, "approval", {"token": token, "approved": approved}, request_id)

    def resume(self, task_id, request_id):
        return self._enqueue(task_id, "resume", {}, request_id)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("任务服务已启动")
        self._lease = TaskFileLock(self.settings.db_path, "web-service-worker")
        try:
            self._lease.__enter__()
        except BaseException:
            self._lease = None
            raise
        try:
            with self._db() as conn, conn:
                conn.execute(
                    "UPDATE web_jobs SET status='interrupted',error=?,updated_at=? WHERE status='running'",
                    ("服务中断；请检查状态后手动恢复", now()),
                )
            self._stop.clear()
            self._worker_error = ""
            self._thread = threading.Thread(target=self._worker, name="research-web-worker", daemon=True)
            self._thread.start()
            self._wake.set()
        except BaseException:
            self._lease.__exit__(None, None, None)
            self._lease = None
            raise

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                # 仍在外部调用中时不能释放单 worker 锁，避免另一实例重复执行。
                raise RuntimeError("研究调用仍在结束中，请等待当前调用超时后再关闭")
            self._thread = None
        if self._lease is not None:
            self._lease.__exit__(None, None, None)
            self._lease = None

    def _claim(self):
        with self._db() as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM web_jobs WHERE status='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is not None:
                conn.execute("UPDATE web_jobs SET status='running',updated_at=? WHERE id=?", (now(), row["id"]))
            return dict(row) if row is not None else None

    def _worker(self):
        while not self._stop.is_set():
            try:
                job = self._claim()
            except Exception:
                logger.exception("读取网页任务队列失败")
                self._worker_error = "任务队列暂时不可读，正在等待恢复"
                self._stop.wait(1)
                continue
            self._worker_error = ""
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            status, error = "done", ""
            try:
                self._execute(job)
            except (WorkerStopped, TaskBusy) as failure:
                status, error = "interrupted", str(failure)
            except Exception as failure:
                logger.exception("网页研究任务执行失败：%s", job["task_id"])
                status, error = "failed", type(failure).__name__ + ": " + str(failure)[:1000]
            # 存储暂时不可写时保留本次结果并重试，不重跑模型，也不让 worker 静默退出。
            while True:
                try:
                    with self._db(timeout=1) as conn, conn:
                        conn.execute("UPDATE web_jobs SET status=?,error=?,updated_at=? WHERE id=?",
                                     (status, error, now(), job["id"]))
                    self._worker_error = ""
                    break
                except sqlite3.Error:
                    if not self._worker_error:
                        logger.exception("保存网页任务结果失败，等待数据库恢复")
                    self._worker_error = "任务结果暂时无法保存，正在等待数据库恢复"
                    if self._stop.wait(1):
                        return

    def health(self):
        healthy = bool(self._thread is not None and self._thread.is_alive() and not self._worker_error)
        return {"ok": healthy, "error": self._worker_error}

    def _execute(self, job):
        task_id, payload = job["task_id"], json.loads(job["payload"])
        with self.runtime_factory(self.settings) as rt:
            self._check_task(rt.store.conn, task_id)
            if rt.store.get_task(task_id)["status"] == "planning":
                if job["action"] == "approval":
                    raise WebError(409, "任务仍在规划")
                task = rt.agent.plan_existing(task_id)
                if task["status"] != "planned":
                    raise ValueError("；".join(task["errors"]) or "研究规划失败")
            if self._stop.is_set():
                raise WorkerStopped("服务正在关闭；任务已保存，可稍后恢复")
            with rt.controller.session(task_id) as selected:
                info = selected.inspect(task_id)
                if job["action"] == "approval":
                    pending = self._pending(selected, task_id, info)
                    if not pending or pending["token"] != payload["token"]:
                        raise WebError(409, "审批已变化，本次操作未执行")
                    events = selected.resume(task_id, approved=payload["approved"])
                elif info["pending"]:
                    if job["action"] == "create":
                        return
                    raise WebError(409, "任务等待审批，请先批准或拒绝")
                elif not info["exists"]:
                    events = selected.start(task_id)
                elif info["next"]:
                    events = selected.resume(task_id)
                elif job["action"] == "create":
                    return
                else:
                    raise WebError(409, "任务已经结束")
                try:
                    for _ in events:
                        if self._stop.is_set():
                            raise WorkerStopped("服务正在关闭；任务已保存，可稍后恢复")
                finally:
                    close = getattr(events, "close", None)
                    if close is not None:
                        close()
