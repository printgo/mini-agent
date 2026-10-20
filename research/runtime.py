import errno
import hashlib
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import openai
import requests

logger = logging.getLogger(__name__)


class TaskControlError(RuntimeError):
    pass


class TaskBusy(TaskControlError):
    pass


class TaskCancelled(TaskControlError):
    pass


class TaskStateError(TaskControlError):
    pass


class TaskFileLock:
    """同一数据库、同一任务使用同一个 OS 文件锁。"""

    def __init__(self, db_path, task_id):
        database = Path(db_path).resolve()
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        self.path = database.with_name(
            f"{database.name}-task-{digest}.lock"
        )
        self.fd = None

    def __enter__(self):
        if self.fd is not None:
            raise RuntimeError("同一个锁对象不能重复进入")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.set_inheritable(fd, False)
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise TaskBusy("任务正在被其他命令处理，请稍后重试") from error
            raise
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return self

    def __exit__(self, exc_type, exc, traceback):
        fd, self.fd = self.fd, None
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        # 保留锁文件，避免删除后其他进程锁住不同的文件对象。


def failure_category(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__

    for item in reversed(chain):
        if isinstance(item, (KeyboardInterrupt, EOFError)):
            return "interrupted"
        if isinstance(item, requests.exceptions.SSLError):
            return "configuration"
        if isinstance(item, openai.APIStatusError):
            code = item.status_code
            return "transient" if code in {408, 429} or code >= 500 else "request"
        if isinstance(item, requests.HTTPError):
            code = item.response.status_code if item.response is not None else 0
            return "transient" if code in {408, 429} or code >= 500 else "request"
        if isinstance(item, (
            TimeoutError, ConnectionError, requests.Timeout,
            requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
            openai.APIConnectionError,
        )):
            return "transient"
        if isinstance(item, sqlite3.Error):
            return "storage"
        if isinstance(item, ValueError):
            return "validation"
    return "unknown"


class TaskController:
    def __init__(self, store, research, versions, db_path):
        self.store = store
        self.research = research
        self.versions = versions
        self.db_path = Path(db_path).resolve()
        self.conn = store.conn
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_cancellations (
                task_id TEXT PRIMARY KEY REFERENCES research_tasks(id),
                cancelled_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES research_tasks(id),
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

    def _select(self, task_id):
        version = self.research.checkpoint_version(task_id)
        if version is None:
            return self.research
        if version not in self.versions:
            raise TaskStateError(f"不支持 checkpoint 版本：{version}")
        return self.versions[version]

    def _cancelled_at(self, task_id):
        row = self.conn.execute(
            "SELECT cancelled_at FROM research_cancellations WHERE task_id=?",
            (task_id,),
        ).fetchone()
        return row[0] if row else None

    @contextmanager
    def session(self, task_id):
        # 锁覆盖整个 with，包括惰性 stream 的消费和审批交互。
        with TaskFileLock(self.db_path, task_id):
            self.store.get_task(task_id)
            if self._cancelled_at(task_id):
                raise TaskCancelled("任务已取消，不能恢复")
            selected = self._select(task_id)
            try:
                yield selected
            except TaskControlError:
                raise
            except (Exception, KeyboardInterrupt) as error:
                try:
                    with self.conn:
                        self.conn.execute(
                            "INSERT INTO research_failures "
                            "(task_id, category, message, created_at) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                task_id, failure_category(error),
                                str(error)[:2000] or type(error).__name__,
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        )
                except Exception:
                    logger.exception("保存失败记录失败")
                # 保留原任务状态和 checkpoint，避免覆盖已完成的节点。
                raise

    def inspect(self, task_id):
        selected = self._select(task_id)
        info = selected.inspect(task_id)
        cancelled_at = self._cancelled_at(task_id)
        failures = self.list_failures(task_id)
        info.update(
            version=self.research.checkpoint_version(task_id),
            cancelled_at=cancelled_at,
            resume_allowed=bool(
                not cancelled_at
                and (
                    info["next"]
                    or (not info["exists"] and info["status"] == "planned")
                )
            ),
            last_failure=failures[0] if failures else None,
        )
        return info

    def cancel(self, task_id):
        with TaskFileLock(self.db_path, task_id):
            task = self.store.get_task(task_id)
            if self._cancelled_at(task_id):
                return task
            info = self._select(task_id).inspect(task_id)
            if task["status"] in {"completed", "partial", "cancelled"}:
                raise TaskStateError("任务已结束，不能取消")
            if not (
                (info["exists"] and info["next"])
                or (not info["exists"] and task["status"] == "planned")
            ):
                raise TaskStateError("只能取消 planned 或已暂停的可恢复任务")
            with self.conn:
                self.conn.execute(
                    "INSERT INTO research_cancellations "
                    "(task_id, cancelled_at) VALUES (?, ?)",
                    (task_id, datetime.now(timezone.utc).isoformat()),
                )
                self.conn.execute(
                    "UPDATE research_tasks SET status='cancelled' WHERE id=?",
                    (task_id,),
                )
            return self.store.get_task(task_id)

    def list_failures(self, task_id):
        self.store.get_task(task_id)
        rows = self.conn.execute(
            "SELECT id, category, message, created_at "
            "FROM research_failures WHERE task_id=? ORDER BY id DESC LIMIT 20",
            (task_id,),
        ).fetchall()
        return [
            dict(zip(["id", "category", "message", "created_at"], row))
            for row in rows
        ]