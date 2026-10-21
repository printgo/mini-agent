import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App, { storageKeys } from './App';
import { api, ApiError, type TaskDetail, type TaskSummary } from './api';

vi.mock('./api', async importOriginal => {
  const actual = await importOriginal<typeof import('./api')>();
  return { ...actual, api: {
    listTasks: vi.fn(), getTask: vi.fn(), createTask: vi.fn(),
    approve: vi.fn(), resume: vi.fn(),
  } };
});

const summary = (id: string, question = `问题 ${id}`): TaskSummary => ({
  id, question, status: 'awaiting_approval', created_at: '2026-10-21T09:00:00Z', job_status: null,
});

function detail(id = 'task-a'): TaskDetail {
  return {
    id, question: `问题 ${id}`, created_at: '2026-10-21T09:00:00Z',
    status: 'awaiting_approval', queries: [{ source: 'web', query: 'alpha' }],
    report: '', errors: [], active_job: null, last_job: null, resume_allowed: true,
    pending: { token: 'checkpoint-1', kind: 'initial', queries: [{ source: 'web', query: 'alpha' }],
      round_no: 0, max_rounds: 3, used_supplements: 0, max_supplements: 2 },
    usage: { model_used: 1, model_limit: 12, query_used: 0, query_limit: 8 },
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listTasks).mockResolvedValue({ items: [], total: 0 });
  vi.mocked(api.getTask).mockImplementation(async id => detail(id));
  vi.mocked(api.createTask).mockResolvedValue({ task_id: 'task-a', job_id: 'job-create' });
  vi.mocked(api.approve).mockResolvedValue({ task_id: 'task-a', job_id: 'job-approve' });
  vi.mocked(api.resume).mockResolvedValue({ task_id: 'task-a', job_id: 'job-resume' });
});

describe('研究工作台', () => {
  it('网络失败后重试保持同一创建标识，成功后只选择返回的任务', async () => {
    const user = userEvent.setup();
    vi.mocked(api.createTask).mockRejectedValueOnce(new TypeError('offline'));
    render(<App />);
    await user.type(screen.getByLabelText('你想研究什么？'), 'alpha research');
    await user.click(screen.getByRole('button', { name: '创建研究任务' }));
    await screen.findByRole('alert');
    const original = vi.mocked(api.createTask).mock.calls[0][0];
    expect(original.question).toBe('alpha research');
    expect(original.request_id).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '创建研究任务' }));
    await screen.findByRole('heading', { name: '问题 task-a', level: 1 });
    expect(api.createTask).toHaveBeenCalledTimes(2);
    expect(vi.mocked(api.createTask).mock.calls[1][0]).toEqual(original);
    expect(localStorage.getItem(storageKeys.selected)).toBe('task-a');
    expect(localStorage.getItem(storageKeys.creation)).toBeNull();
  });

  it('丢失创建响应后重新打开页面，可以用原标识继续提交', async () => {
    const user = userEvent.setup();
    vi.mocked(api.createTask).mockRejectedValueOnce(new TypeError('response lost'));
    const first = render(<App />);
    await user.type(screen.getByLabelText('你想研究什么？'), 'persistent submission');
    await user.click(screen.getByRole('button', { name: '创建研究任务' }));
    await screen.findByRole('alert');
    const original = vi.mocked(api.createTask).mock.calls[0][0];
    first.unmount();
    render(<App />);
    expect(screen.getByLabelText('你想研究什么？')).toHaveValue('persistent submission');
    await user.click(screen.getByRole('button', { name: '创建研究任务' }));
    await screen.findByRole('heading', { name: '问题 task-a', level: 1 });
    expect(vi.mocked(api.createTask).mock.calls[1][0]).toEqual(original);
  });

  it('审批携带当前 checkpoint 标识，入队后不允许再次提交', async () => {
    const user = userEvent.setup();
    localStorage.setItem(storageKeys.selected, 'task-a');
    let running = false;
    vi.mocked(api.getTask).mockImplementation(async () => ({
      ...detail(), active_job: running ? { id: 'job-approve', action: 'approve', status: 'queued', error: null } : null,
    }));
    vi.mocked(api.approve).mockImplementation(async () => {
      running = true;
      return { task_id: 'task-a', job_id: 'job-approve' };
    });
    render(<App />);
    const approve = await screen.findByRole('button', { name: '批准并继续' });
    await user.click(approve);
    await waitFor(() => expect(approve).toBeDisabled());
    expect(api.approve).toHaveBeenCalledWith('task-a', {
      approved: true, approval_token: 'checkpoint-1', request_id: expect.any(String),
    });
    await user.click(approve);
    expect(api.approve).toHaveBeenCalledTimes(1);
  });

  it('审批冲突会显示原因并刷新为新的审批状态', async () => {
    const user = userEvent.setup();
    localStorage.setItem(storageKeys.selected, 'task-a');
    vi.mocked(api.approve).mockRejectedValue(new ApiError(409, '审批已变化'));
    render(<App />);
    await user.click(await screen.findByRole('button', { name: '批准并继续' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('审批已变化');
    await waitFor(() => expect(api.getTask).toHaveBeenCalledTimes(2));
    expect(screen.getByRole('alert')).toHaveTextContent('已重新获取任务状态');
  });

  it('恢复已选择的任务并安全显示 Markdown 报告', async () => {
    localStorage.setItem(storageKeys.selected, 'task-completed');
    vi.mocked(api.getTask).mockResolvedValue({
      ...detail('task-completed'), status: 'completed', pending: null, resume_allowed: false,
      report: '# 已完成\n\n**研究结论**\n\n<img src="x" onerror="alert(1)">\n\n[危险](javascript:alert(1))',
    });
    render(<App />);
    await screen.findByRole('heading', { name: '已完成' });
    expect(api.getTask).toHaveBeenCalledWith('task-completed', expect.any(AbortSignal));
    expect(screen.getByText('研究结论').tagName).toBe('STRONG');
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: '危险' })).not.toBeInTheDocument();
  });

  it('切换任务后忽略较晚返回的旧任务响应', async () => {
    const user = userEvent.setup();
    localStorage.setItem(storageKeys.selected, 'task-a');
    vi.mocked(api.listTasks).mockResolvedValue({ items: [summary('task-a'), summary('task-b')], total: 2 });
    let finishOld!: (value: TaskDetail) => void;
    const old = new Promise<TaskDetail>(resolve => { finishOld = resolve; });
    vi.mocked(api.getTask).mockImplementation(id => id === 'task-a' ? old : Promise.resolve(detail('task-b')));
    render(<App />);
    await user.click(await screen.findByRole('button', { name: /问题 task-b/ }));
    await screen.findByRole('heading', { name: '问题 task-b', level: 1 });
    await act(async () => { finishOld(detail('task-a')); await old; });
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('问题 task-b');
  });

  it('卸载页面时停止轮询并取消正在读取的请求', async () => {
    vi.mocked(api.listTasks).mockReturnValue(new Promise(() => {}));
    const view = render(<App />);
    const signal = vi.mocked(api.listTasks).mock.calls[0][1]!;
    expect(signal.aborted).toBe(false);
    view.unmount();
    expect(signal.aborted).toBe(true);
  });

  it('任务列表识别中断和失败，但已完成状态优先于过期的后台标记', async () => {
    vi.mocked(api.listTasks).mockResolvedValue({ items: [
      { ...summary('interrupted'), status: 'planning', job_status: 'interrupted' },
      { ...summary('failed'), status: 'writing', job_status: 'failed' },
      { ...summary('completed'), status: 'completed', job_status: 'interrupted' },
      { ...summary('partial'), status: 'partial', job_status: 'failed' },
    ], total: 4 });
    render(<App />);
    const interrupted = await screen.findByRole('button', { name: /问题 interrupted/ });
    expect(within(interrupted).getByText('执行已中断')).toBeInTheDocument();
    expect(within(screen.getByRole('button', { name: /问题 failed/ })).getByText('执行失败')).toBeInTheDocument();
    expect(within(screen.getByRole('button', { name: /问题 completed/ })).getByText('研究完成')).toBeInTheDocument();
    expect(within(screen.getByRole('button', { name: /问题 partial/ })).getByText('部分完成')).toBeInTheDocument();
  });
});
