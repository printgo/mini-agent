import { useEffect, useRef, useState, type FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { api, ApiError, type Accepted, type Creation, type TaskDetail, type TaskSummary } from './api';

export const storageKeys = {
  selected: 'research-workbench:selected',
  creation: 'research-workbench:creation',
};

const labels: Record<string, string> = {
  planning: '正在规划', planned: '计划已生成', awaiting_approval: '等待审批',
  collecting: '正在采集', retrieving: '正在检索', writing: '正在写作',
  reviewing: '正在审查', completed: '研究完成', partial: '部分完成',
  failed: '执行失败', cancelled: '已取消', queued: '排队中',
  running: '正在执行', done: '本次执行结束', interrupted: '执行已中断',
};

const sourceNames: Record<string, string> = {
  web: '网页', paper: '论文摘要', github: 'GitHub', local: '本地文档',
};

function saved(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

function remember(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* 页面仍可使用，当前标签页中的请求标识继续保留。 */ }
}

function savedCreation(): Creation | null {
  try {
    const value: unknown = JSON.parse(saved(storageKeys.creation) ?? 'null');
    if (!value || typeof value !== 'object') return null;
    const candidate = value as Partial<Creation>;
    return typeof candidate.question === 'string' && candidate.question.trim().length > 0
      && candidate.question.length <= 1000 && typeof candidate.request_id === 'string'
      && candidate.request_id.length > 0 && candidate.request_id.length <= 128
      ? candidate as Creation : null;
  } catch { return null; }
}

function failure(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.name === 'AbortError') return '请求超时，已停止等待服务响应。';
  return '无法连接本地服务，请确认后端已启动。';
}

function timestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

function taskStatus(task: TaskSummary | TaskDetail): string {
  if (['completed', 'partial', 'cancelled'].includes(task.status)) return task.status;
  const job = 'active_job' in task
    ? task.active_job?.status ?? task.last_job?.status : task.job_status;
  if (job === 'queued') return 'queued';
  if (job === 'failed' || job === 'interrupted') return job;
  return task.status;
}

export default function App() {
  const creation = useRef<Creation | null>(savedCreation());
  const actionIds = useRef(new Map<string, string>());
  const mounted = useRef(true);
  const [question, setQuestion] = useState(creation.current?.question ?? '');
  const [selectedId, setSelectedId] = useState<string | null>(() => saved(storageKeys.selected));
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [operation, setOperation] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [connection, setConnection] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      controller = new AbortController();
      timeout = setTimeout(() => controller?.abort(), 15000);
      const problems: string[] = [];
      // 两个区域各自在响应后更新，慢详情不会阻塞历史任务导航。
      const listRequest = api.listTasks(page * 50, controller.signal).then(list => {
        if (stopped) return;
        setTasks(list.items);
        setTotal(list.total);
        if (page > 0 && page * 50 >= list.total) {
          setPage(Math.max(0, Math.ceil(list.total / 50) - 1));
        }
      }).catch(error => { if (!stopped) problems.push(failure(error)); });
      const detailRequest = selectedId ? api.getTask(selectedId, controller.signal).then(item => {
        if (!stopped) setDetail(item);
      }).catch(error => { if (!stopped) problems.push(failure(error)); }) : Promise.resolve();
      if (!selectedId) setDetail(null);
      await Promise.all([listRequest, detailRequest]);
      if (timeout !== undefined) clearTimeout(timeout);
      if (stopped) return;
      setPollError(problems.length ? [...new Set(problems)].join(' ') : null);
      setConnection(problems.length ? 'error' : 'ready');
      // 上一轮结束后再安排下一轮，避免慢请求堆叠。
      timer = setTimeout(() => { void poll(); }, 2000);
    }

    void poll();
    return () => {
      stopped = true;
      controller?.abort();
      if (timer !== undefined) clearTimeout(timer);
      if (timeout !== undefined) clearTimeout(timeout);
    };
  }, [selectedId, page, refresh]);

  function selectTask(id: string | null) {
    remember(storageKeys.selected, id);
    setSelectedId(id);
    setActionError(null);
  }

  function idFor(action: string): string {
    let value = actionIds.current.get(action);
    if (!value) {
      value = crypto.randomUUID();
      actionIds.current.set(action, value);
    }
    return value;
  }

  async function create(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || operation !== null) return;
    const body = creation.current?.question === text ? creation.current : {
      question: text, request_id: crypto.randomUUID(),
    };
    creation.current = body;
    remember(storageKeys.creation, JSON.stringify(body));
    setOperation('create');
    setActionError(null);
    try {
      const accepted = await api.createTask(body);
      if (!mounted.current) return;
      creation.current = null;
      remember(storageKeys.creation, null);
      setQuestion('');
      selectTask(accepted.task_id);
      setPage(0);
      setRefresh(value => value + 1);
    } catch (error) {
      if (mounted.current) setActionError(failure(error));
    } finally {
      if (mounted.current) setOperation(null);
    }
  }

  async function taskAction(approved?: boolean) {
    const task = detail?.id === selectedId ? detail : null;
    if (!task || task.active_job || operation !== null) return;
    if (approved !== undefined && !task.pending) return;
    const action = approved === undefined ? 'resume' : approved ? 'approve' : 'reject';
    const key = `${task.id}:${action}:${task.pending?.token ?? ''}`;
    setOperation(action);
    setActionError(null);
    try {
      let accepted: Accepted;
      if (approved === undefined) accepted = await api.resume(task.id, idFor(key));
      else accepted = await api.approve(task.id, {
        approved, approval_token: task.pending!.token, request_id: idFor(key),
      });
      if (!mounted.current) return;
      actionIds.current.delete(key);
      setDetail(value => value?.id === task.id ? {
        ...value, active_job: { id: accepted.job_id, action, status: 'queued', error: null },
      } : value);
      setRefresh(value => value + 1);
    } catch (error) {
      if (!mounted.current) return;
      if (error instanceof ApiError && error.status === 409) {
        setActionError(`任务状态已变化：${error.message}。已重新获取任务状态。`);
        setRefresh(value => value + 1);
      } else setActionError(failure(error));
    } finally {
      if (mounted.current) setOperation(null);
    }
  }

  const task = detail?.id === selectedId ? detail : null;
  const busy = operation !== null;
  const active = Boolean(task?.active_job);
  const state = task ? taskStatus(task) : '';

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="研究任务导航">
        <a className="brand" href="#" onClick={event => { event.preventDefault(); if (!busy) selectTask(null); }}>
          <span className="brand-mark" aria-hidden="true">R</span>
          <span>Research Agent<small>研究工作台</small></span>
        </a>
        <button className="button new-task" disabled={busy} onClick={() => selectTask(null)}>
          <span aria-hidden="true">＋</span> 新建研究
        </button>
        <div className="sidebar-heading"><h2>研究任务</h2><span>{total}</span></div>
        <nav className="task-list" aria-label="历史研究">
          {tasks.map(item => (
            <button key={item.id} className={`task-item ${selectedId === item.id ? 'selected' : ''}`}
              disabled={busy} aria-current={selectedId === item.id ? 'page' : undefined}
              onClick={() => selectTask(item.id)}>
              <span className="task-question">{item.question}</span>
              <span className="task-meta"><span>{labels[taskStatus(item)] ?? item.status}</span>
                <time dateTime={item.created_at}>{timestamp(item.created_at)}</time></span>
            </button>
          ))}
          {tasks.length === 0 && <p className="sidebar-empty">研究任务将保存在这里。</p>}
        </nav>
        {total > 50 && <div className="pagination">
          <button disabled={page === 0 || busy} onClick={() => setPage(value => value - 1)}>上一页</button>
          <span>{page + 1} / {Math.ceil(total / 50)}</span>
          <button disabled={(page + 1) * 50 >= total || busy} onClick={() => setPage(value => value + 1)}>下一页</button>
        </div>}
        <div className="sidebar-footer">规划 · 采集 · 检索 · 写作 · 审查</div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <span>研究工作台</span>
          <span className={`connection ${connection}`} role="status">
            <span className="connection-dot" aria-hidden="true" />
            {connection === 'ready' ? '本地服务已连接' : connection === 'loading' ? '正在连接服务' : '连接待恢复'}
          </span>
        </header>
        <div className="workspace-content">
          {pollError && <div className="notice warning" role="status">{pollError} 页面会自动重新连接。</div>}
          {actionError && <div className="notice error" role="alert">{actionError}</div>}

          {!selectedId ? <section className="compose-view">
            <div className="eyebrow">从一个问题开始</div>
            <h1>让研究有据可循。</h1>
            <p className="intro">描述目标，审阅检索计划，再查看带有来源引用的研究报告。</p>
            <form className="card compose-card" onSubmit={event => { void create(event); }}>
              <label htmlFor="research-question">你想研究什么？</label>
              <textarea id="research-question" value={question} maxLength={1000}
                disabled={busy} rows={7} required
                placeholder="例如：比较两种 Agent 中断恢复方式，说明各自适用的场景并引用依据。"
                onChange={event => setQuestion(event.target.value)} />
              <div className="compose-footer"><span>{question.length} / 1000</span>
                <button className="button primary" type="submit" disabled={busy || !question.trim()}>
                  {operation === 'create' ? '正在提交…' : '创建研究任务'}
                </button></div>
              <p className="form-hint">提交后先生成计划，采集将在你批准后开始。网络中断时可重试同一次提交。</p>
            </form>
            <div className="feature-grid">
              <div><span>01</span><h2>先规划，再执行</h2><p>检索查询清晰展示，由你决定是否继续。</p></div>
              <div><span>02</span><h2>保留任务状态</h2><p>切换页面或刷新后，可以继续查看同一任务。</p></div>
              <div><span>03</span><h2>报告关联来源</h2><p>阅读研究结论时，同时看到文中的证据引用。</p></div>
            </div>
          </section> : task ? <section className="task-view">
            <div className="task-heading"><div><div className="eyebrow">研究详情</div><h1>{task.question}</h1></div>
              <span className={`badge ${state}`}>{labels[state] ?? state}</span></div>
            <div className="detail-meta"><span>创建于 {timestamp(task.created_at)}</span>
              <code title={task.id}>任务 {task.id.slice(0, 12)}</code>
              <button className="text-button" disabled={busy} onClick={() => setRefresh(value => value + 1)}>刷新状态</button></div>

            <div className="metrics" aria-label="调用用量">
              <div><span>模型逻辑调用</span><strong>{task.usage.model_used}<small> / {task.usage.model_limit}</small></strong></div>
              <div><span>远程采集查询</span><strong>{task.usage.query_used}<small> / {task.usage.query_limit}</small></strong></div>
              <div><span>执行阶段</span><strong className="stage-name">{labels[task.status] ?? task.status}</strong></div>
            </div>

            {task.last_job?.error && !active && <div className="notice error" role="alert">
              <strong>{task.last_job.status === 'interrupted' ? '本次执行已中断。' : '本次执行未完成。'}</strong>
              <span>{task.last_job.error}</span>
            </div>}
            {active && <div className="notice info" role="status">
              <span className="spinner" aria-hidden="true" />
              {task.active_job!.status === 'queued' ? '任务已进入队列。' : '后台正在处理任务。'}
              你可以切换任务或稍后返回查看。
            </div>}

            {task.pending && <section className="card approval-card" aria-label="检索审批">
              <div className="card-heading"><div><div className="eyebrow">需要你的决定</div>
                <h2>{task.pending.kind === 'supplement' ? '补充检索审批' : '初始检索审批'}</h2></div>
                <span className="badge awaiting_approval">待确认</span></div>
              <ol className="query-list">{task.pending.queries.map((query, index) => <li key={`${query.source}:${index}`}>
                <span className="source-tag">{sourceNames[query.source] ?? query.source}</span><span>{query.query}</span>
              </li>)}</ol>
              <p className="form-hint">已完成 {task.pending.round_no} / {task.pending.max_rounds} 轮，
                已使用 {task.pending.used_supplements} / {task.pending.max_supplements} 项补查。</p>
              <div className="actions"><button className="button primary" disabled={busy || active}
                onClick={() => { void taskAction(true); }}>批准并继续</button>
                <button className="button secondary" disabled={busy || active}
                  onClick={() => { void taskAction(false); }}>拒绝本次检索</button></div>
            </section>}

            {task.resume_allowed && !task.pending && !active && <div className="resume-row">
              <p>已保存任务状态，可以从待处理步骤继续。</p>
              <button className="button primary" disabled={busy} onClick={() => { void taskAction(); }}>
                {operation === 'resume' ? '正在提交…' : '恢复任务'}
              </button></div>}

            <section className="card report-card" aria-label="研究报告">
              <div className="card-heading"><h2>研究报告</h2><span className="subtle">保留来源引用</span></div>
              {task.report ? <article className="markdown">
                <ReactMarkdown skipHtml urlTransform={url => /^https?:\/\//i.test(url) ? url : ''}
                  components={{ a: ({ href, children }) => href
                    ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
                    : <span>{children}</span> }}>{task.report}</ReactMarkdown>
              </article> : <div className="report-empty"><span aria-hidden="true">≡</span>
                <h3>报告将在执行结束后显示</h3><p>研究会依次完成采集、检索、写作与审查。</p></div>}
            </section>
            {task.errors.length > 0 && <details className="card diagnostics">
              <summary>任务提示与采集记录（{task.errors.length}）</summary>
              <ul>{task.errors.map((error, index) => <li key={index}>{error}</li>)}</ul>
            </details>}
          </section> : <section className="empty-state"><h1>正在读取任务</h1>
            <p>任务状态由本地服务保存，连接恢复后会继续显示。</p>
            <button className="button secondary" onClick={() => selectTask(null)}>返回新建研究</button></section>}
        </div>
      </main>
    </div>
  );
}
