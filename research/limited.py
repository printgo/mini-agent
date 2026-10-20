from contextlib import contextmanager
from contextvars import ContextVar

from research.generation import GenerationWorkflowAgent


_MODEL_TASK = ContextVar("research_model_task", default=None)


class LimitedWorkflowAgent(GenerationWorkflowAgent):
    VERSION = "16.0"

    def __init__(self, store, checkpointer, ledger):
        self.ledger = ledger
        super().__init__(store, checkpointer)

    @contextmanager
    def _model_scope(self, task_id):
        token = _MODEL_TASK.set((self, task_id))
        try:
            yield
        finally:
            _MODEL_TASK.reset(token)

    def _ask_json(self, instruction, data):
        scope = _MODEL_TASK.get()
        if scope is not None and scope[0] is self:
            self.ledger.reserve(scope[1], "model")
        return super()._ask_json(instruction, data)

    def _node_write(self, state):
        with self._model_scope(state["task_id"]):
            return super()._node_write(state)

    def _node_review(self, state):
        with self._model_scope(state["task_id"]):
            return super()._node_review(state)

    def _collect(self, task_id, queries, errors):
        for entry in queries:
            if self.store.get_collection(
                task_id, entry["source"], entry["query"]
            ) is None:
                self.ledger.reserve(task_id, "query")
            super()._collect(task_id, [entry], errors)