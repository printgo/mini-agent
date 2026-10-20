import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from research.runtime import (
    TaskBusy, TaskCancelled, TaskController, TaskFileLock,
    TaskStateError, failure_category,
)


WORKER = """
import os
import sys
from research.runtime import TaskBusy, TaskFileLock
try:
    with TaskFileLock(sys.argv[1], sys.argv[2]):
        print("locked", flush=True)
        if sys.stdin.readline().strip() == "crash":
            os._exit(73)
except TaskBusy:
    print("busy", flush=True)
"""


def wait_for_lock_release(db_path, task_id, timeout=5.0, interval=0.02):
    """等待实际锁可用；只重试 TaskBusy，超时仍判失败。"""
    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        attempts += 1
        try:
            with TaskFileLock(db_path, task_id):
                return attempts
        except TaskBusy as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"任务 {task_id} 在 {timeout} 秒内仍未释放锁"
                ) from error
            time.sleep(min(interval, remaining))


class FakeStore:
    def __init__(self, conn):
        self.conn = conn
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_tasks "
                "(id TEXT PRIMARY KEY, status TEXT NOT NULL)"
            )

    def add(self, task_id, status="planned"):
        with self.conn:
            self.conn.execute(
                "INSERT INTO research_tasks VALUES (?, ?)", (task_id, status)
            )

    def get_task(self, task_id):
        row = self.conn.execute(
            "SELECT id, status FROM research_tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return {"id": row[0], "status": row[1]}


class FakeAgent:
    def __init__(self, version, store, checkpoints):
        self.VERSION = version
        self.store = store
        self.checkpoints = checkpoints

    def checkpoint_version(self, task_id):
        self.store.get_task(task_id)
        return self.checkpoints.get(task_id)

    def inspect(self, task_id):
        task = self.store.get_task(task_id)
        exists = task_id in self.checkpoints
        return {
            "status": task["status"], "exists": exists,
            "next": ["collect"] if exists else [],
            "pending": [], "node_errors": [],
        }


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "research.db"

    @staticmethod
    def stop(process):
        try:
            process.communicate(input="\n", timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)

    def worker(self, task_id):
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", WORKER, str(self.db_path), task_id],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        self.addCleanup(self.stop, process)
        messages = queue.Queue()
        threading.Thread(
            target=lambda: messages.put(process.stdout.readline().strip()),
            daemon=True,
        ).start()
        return process, messages.get(timeout=15)

    def test_same_task_is_exclusive_across_processes(self):
        _, first = self.worker("same")
        _, second = self.worker("same")
        self.assertEqual((first, second), ("locked", "busy"))

    def test_different_tasks_can_run_concurrently(self):
        _, first = self.worker("first")
        _, second = self.worker("second")
        self.assertEqual((first, second), ("locked", "locked"))

    def test_forced_exit_releases_lock(self):
        process, message = self.worker("crashed")
        self.assertEqual(message, "locked")

        # 让真正持锁的解释器直接退出，跳过 with 的正常解锁。
        process.stdin.write("crash\n")
        process.stdin.flush()
        self.assertEqual(process.wait(timeout=10), 73)

        wait_for_lock_release(self.db_path, "crashed")

    def test_exception_releases_lock_and_retains_file(self):
        with self.assertRaises(RuntimeError):
            with TaskFileLock(self.db_path, "exception"):
                raise RuntimeError("injected")
        files = list(Path(self.tmp.name).glob("*.lock"))
        self.assertEqual(len(files), 1)
        with TaskFileLock(self.db_path, "exception"):
            pass
        self.assertEqual(files, list(Path(self.tmp.name).glob("*.lock")))

    def test_wait_retries_temporary_busy(self):
        with patch(__name__ + ".TaskFileLock") as factory:
            context = factory.return_value
            context.__enter__.side_effect = [
                TaskBusy("cleanup pending"),
                TaskBusy("cleanup pending"),
                context,
            ]
            attempts = wait_for_lock_release(self.db_path, "delayed")
            self.assertEqual(attempts, 3)

    def test_wait_times_out_while_owner_is_alive(self):
        process, message = self.worker("alive")
        self.assertEqual(message, "locked")
        with self.assertRaises(TimeoutError):
            wait_for_lock_release(
                self.db_path, "alive", timeout=0.1, interval=0.01
            )
        self.assertIsNone(process.poll())

    def test_wait_does_not_hide_other_errors(self):
        with patch(__name__ + ".TaskFileLock") as factory:
            factory.return_value.__enter__.side_effect = OSError("bad handle")
            with self.assertRaisesRegex(OSError, "bad handle"):
                wait_for_lock_release(self.db_path, "invalid")


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "research.db"
        self.conn = sqlite3.connect(self.db_path)
        self.addCleanup(self.conn.close)
        self.store = FakeStore(self.conn)
        self.checkpoints = {}
        self.agents = {
            version: FakeAgent(version, self.store, self.checkpoints)
            for version in ("12.0", "13.0", "14.0", "15.0")
        }
        self.current = self.agents["15.0"]
        self.controller = TaskController(
            self.store, self.current, self.agents, self.db_path
        )

    def test_old_and_new_task_routing(self):
        for version in ("12.0", "13.0", "14.0"):
            self.store.add(version, "collecting")
            self.checkpoints[version] = version
            with self.controller.session(version) as selected:
                self.assertIs(selected, self.agents[version])
        self.store.add("new")
        with self.controller.session("new") as selected:
            self.assertIs(selected, self.current)

    def test_failure_is_recorded_and_retry_is_possible(self):
        self.store.add("failure", "collecting")
        self.checkpoints["failure"] = "15.0"
        with self.assertRaises(TimeoutError):
            with self.controller.session("failure"):
                raise TimeoutError("injected timeout")
        failures = self.controller.list_failures("failure")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["category"], "transient")
        self.assertEqual(self.store.get_task("failure")["status"], "collecting")
        with self.controller.session("failure"):
            pass

    def test_busy_does_not_record_failure_or_cancel(self):
        self.store.add("busy")
        with TaskFileLock(self.db_path, "busy"):
            with self.assertRaises(TaskBusy):
                with self.controller.session("busy"):
                    self.fail("must not enter")
            with self.assertRaises(TaskBusy):
                self.controller.cancel("busy")
        self.assertEqual(self.controller.list_failures("busy"), [])
        self.assertEqual(self.store.get_task("busy")["status"], "planned")

    def test_cancelled_checkpoint_cannot_resume(self):
        self.store.add("cancelled", "collecting")
        self.checkpoints["cancelled"] = "14.0"
        self.controller.cancel("cancelled")
        info = self.controller.inspect("cancelled")
        self.assertTrue(info["next"])
        self.assertFalse(info["resume_allowed"])
        self.assertIsNotNone(info["cancelled_at"])
        with self.assertRaises(TaskCancelled):
            with self.controller.session("cancelled"):
                self.fail("must not enter")
        self.assertEqual(
            self.controller.cancel("cancelled")["status"], "cancelled"
        )

    def test_cancellation_survives_new_connection(self):
        self.store.add("persistent")
        self.controller.cancel("persistent")
        with closing(sqlite3.connect(self.db_path)) as connection:
            other_store = FakeStore(connection)
            current = FakeAgent("15.0", other_store, self.checkpoints)
            other = TaskController(
                other_store, current, {"15.0": current}, self.db_path
            )
            with self.assertRaises(TaskCancelled):
                with other.session("persistent"):
                    self.fail("must not enter")

    def test_planning_and_terminal_tasks_cannot_be_cancelled(self):
        for status in ("planning", "completed", "partial", "cancelled"):
            self.store.add(status, status)
            with self.assertRaises(TaskStateError):
                self.controller.cancel(status)
            self.assertEqual(self.store.get_task(status)["status"], status)

    def test_cancel_is_atomic(self):
        self.store.add("rollback")
        self.conn.execute("""
            CREATE TRIGGER reject_cancel BEFORE UPDATE ON research_tasks
            WHEN NEW.status='cancelled'
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        with self.assertRaises(sqlite3.DatabaseError):
            self.controller.cancel("rollback")
        self.assertEqual(self.store.get_task("rollback")["status"], "planned")
        self.assertIsNone(self.controller.inspect("rollback")["cancelled_at"])
        with TaskFileLock(self.db_path, "rollback"):
            pass

    def test_wrapped_error_classification(self):
        error = RuntimeError("wrapped")
        error.__cause__ = TimeoutError("timeout")
        self.assertEqual(failure_category(error), "transient")
        self.assertEqual(failure_category(ValueError("bad")), "validation")
        self.assertEqual(
            failure_category(sqlite3.OperationalError("locked")), "storage"
        )
        self.assertEqual(
            failure_category(KeyboardInterrupt()), "interrupted"
        )


if __name__ == "__main__":
    unittest.main()