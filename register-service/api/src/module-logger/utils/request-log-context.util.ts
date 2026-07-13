import { AsyncLocalStorage } from 'async_hooks';
import type { RequestLogContext } from '@/module-logger/interfaces/logger.interfaces';

const requestLogContextStorage = new AsyncLocalStorage<RequestLogContext>();

export function runWithRequestLogContext<T>(context: RequestLogContext, callback: () => T): T {
    return requestLogContextStorage.run(context, callback);
}

export function getRequestLogContext(): RequestLogContext | undefined {
    return requestLogContextStorage.getStore();
}
