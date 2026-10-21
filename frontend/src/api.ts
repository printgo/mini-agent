export interface Query {
  source: string;
  query: string;
}

export interface Job {
  id: string;
  action: string;
  status: string;
  error: string | null;
}

export interface TaskSummary {
  id: string;
  question: string;
  status: string;
  created_at: string;
  job_status: string | null;
}

export interface TaskDetail extends Omit<TaskSummary, 'job_status'> {
  queries: Query[];
  errors: string[];
  report: string;
  active_job: Job | null;
  last_job: Job | null;
  pending: {
    token: string;
    kind: string;
    queries: Query[];
    round_no: number;
    max_rounds: number;
    used_supplements: number;
    max_supplements: number;
  } | null;
  resume_allowed: boolean;
  usage: {
    model_used: number;
    model_limit: number;
    query_used: number;
    query_limit: number;
  };
}

export interface Accepted {
  task_id: string;
  job_id: string;
}

export interface Creation {
  question: string;
  request_id: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

function errorText(body: unknown, fallback: string): string {
  if (!body || typeof body !== 'object' || !('detail' in body)) return fallback;
  const detail = body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((row: unknown) => {
      if (!row || typeof row !== 'object') return '请求参数无效';
      const value = row as { loc?: unknown; msg?: unknown };
      const location = Array.isArray(value.loc) ? value.loc.join('.') : '';
      return `${location ? `${location}：` : ''}${String(value.msg ?? '请求参数无效')}`;
    }).join('；');
  }
  return fallback;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  // 轮询使用调用方的生命周期信号；提交操作使用有界超时，超时后可按原标识重试。
  const controller = options.signal ? null : new AbortController();
  const timeout = controller ? setTimeout(() => controller.abort(), 15000) : undefined;
  try {
    const response = await fetch(`/api${path}`, {
      ...options,
      signal: options.signal ?? controller?.signal,
      cache: 'no-store',
      headers: { Accept: 'application/json', ...options.headers },
    });
    const text = await response.text();
    let body: unknown;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      throw new ApiError(response.status, '服务返回了非 JSON 内容，请检查后端是否已启动。');
    }
    if (!response.ok) {
      throw new ApiError(response.status, errorText(body, `请求失败（HTTP ${response.status}）`));
    }
    return body as T;
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export const api = {
  listTasks: (offset: number, signal?: AbortSignal) => request<{
    items: TaskSummary[];
    total: number;
  }>(`/tasks?limit=50&offset=${offset}`, { signal }),
  getTask: (id: string, signal?: AbortSignal) => request<TaskDetail>(
    `/tasks/${encodeURIComponent(id)}`, { signal },
  ),
  createTask: (body: Creation) => post<Accepted>('/tasks', body),
  approve: (id: string, body: {
    approved: boolean;
    approval_token: string;
    request_id: string;
  }) => post<Accepted>(`/tasks/${encodeURIComponent(id)}/approval`, body),
  resume: (id: string, requestId: string) => post<Accepted>(
    `/tasks/${encodeURIComponent(id)}/resume`, { request_id: requestId },
  ),
};
