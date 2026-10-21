from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from starlette.concurrency import run_in_threadpool

from research.web_service import TaskService, WebError, WebSettings


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=128)


class CreateTask(RequestBody):
    question: str = Field(min_length=1, max_length=1000)


class Approval(RequestBody):
    approved: StrictBool
    approval_token: str = Field(min_length=1, max_length=256)


def create_app(settings=None, service=None):
    service = service or TaskService(settings or WebSettings())

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(service.start)
        try:
            yield
        finally:
            await run_in_threadpool(service.stop)

    app = FastAPI(title="Research Agent", lifespan=lifespan)
    app.state.task_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @app.exception_handler(WebError)
    async def web_error(request: Request, error: WebError):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.get("/api/health")
    def health():
        result = service.health()
        return JSONResponse(result, status_code=200 if result["ok"] else 503)

    @app.post("/api/tasks", status_code=202)
    def create(body: CreateTask):
        return service.create(body.question, body.request_id)

    @app.get("/api/tasks")
    def tasks(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        return service.list_tasks(limit, offset)

    @app.get("/api/tasks/{task_id}")
    def detail(task_id: str):
        return service.detail(task_id)

    @app.post("/api/tasks/{task_id}/approval", status_code=202)
    def approve(task_id: str, body: Approval):
        return service.approve(task_id, body.approval_token, body.approved, body.request_id)

    @app.post("/api/tasks/{task_id}/resume", status_code=202)
    def resume(task_id: str, body: RequestBody):
        return service.resume(task_id, body.request_id)

    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        return JSONResponse({
            "message": "前端尚未构建。请在 frontend 执行 npm install 和 npm run build，"
                       "或启动 npm run dev 后打开 http://127.0.0.1:5173。"
        }, status_code=503)

    return app
