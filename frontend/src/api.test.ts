import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from './api';

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('API 请求边界', () => {
  it('将表单校验错误转换成可读消息', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [
      { loc: ['body', 'question'], msg: 'String should have at least 1 character' },
    ] }), { status: 422 })));
    await expect(api.createTask({ question: '', request_id: 'r1' })).rejects.toThrow('body.question');
  });

  it('保留冲突状态供审批页面处理', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '状态已变化' }), { status: 409 })));
    await expect(api.approve('task-a', {
      approved: true, approval_token: 'cp1', request_id: 'r2',
    })).rejects.toMatchObject({ status: 409, message: '状态已变化' });
  });

  it('后端代理未连接时不将 HTML 当成研究结果', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>proxy unavailable</html>', { status: 502 })));
    await expect(api.listTasks(0)).rejects.toBeInstanceOf(ApiError);
  });

  it('提交请求在15秒后中止，避免页面永久处于提交状态', async () => {
    vi.useFakeTimers();
    let requestSignal: AbortSignal | null | undefined;
    vi.stubGlobal('fetch', vi.fn((_input: RequestInfo | URL, options?: RequestInit) => {
      requestSignal = options?.signal;
      return new Promise((_resolve, reject) => {
        requestSignal!.addEventListener('abort', () => reject(new DOMException('Timed out', 'AbortError')));
      });
    }));
    const request = api.createTask({ question: 'alpha', request_id: 'retry-preserved' });
    const expected = expect(request).rejects.toMatchObject({ name: 'AbortError' });
    await vi.advanceTimersByTimeAsync(15000);
    await expected;
    expect(requestSignal?.aborted).toBe(true);
  });

  it('轮询保留调用方的AbortSignal，不另行覆盖生命周期', async () => {
    const controller = new AbortController();
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [], total: 0 })));
    vi.stubGlobal('fetch', fetcher);
    await api.listTasks(0, controller.signal);
    expect(fetcher).toHaveBeenCalledWith('/api/tasks?limit=50&offset=0', expect.objectContaining({ signal: controller.signal }));
  });
});
