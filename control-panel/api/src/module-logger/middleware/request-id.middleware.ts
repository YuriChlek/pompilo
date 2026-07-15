import { randomUUID } from 'crypto';
import type { NextFunction, Request, Response } from 'express';
import type { LoggerConfig } from '@/module-logger/interfaces/logger.interfaces';
import { runWithRequestLogContext } from '@/module-logger/utils/request-log-context.util';

const MAX_REQUEST_ID_LENGTH = 200;
const VALID_REQUEST_ID_VALUE_PATTERN = /^[\x20-\x7e]+$/;

export function createRequestIdMiddleware(config: Pick<LoggerConfig, 'requestIdHeader'>) {
    const requestIdHeader = config.requestIdHeader.toLowerCase();

    return (request: Request, response: Response, next: NextFunction): void => {
        const requestId = resolveRequestId(request.headers[requestIdHeader]) ?? randomUUID();

        response.setHeader(config.requestIdHeader, requestId);
        runWithRequestLogContext({ requestId }, next);
    };
}

function resolveRequestId(rawHeader: string | string[] | undefined): string | undefined {
    const rawValue = Array.isArray(rawHeader) ? rawHeader[0] : rawHeader;
    const requestId = rawValue?.trim();

    if (!requestId) {
        return undefined;
    }

    if (
        requestId.length > MAX_REQUEST_ID_LENGTH ||
        !VALID_REQUEST_ID_VALUE_PATTERN.test(requestId)
    ) {
        return undefined;
    }

    return requestId;
}
