import { Injectable } from '@nestjs/common';
import {
    LOGGER_REDACTED_KEYS,
    LOG_REDACTION_CENSOR,
} from '@/module-logger/constants/logger.constants';

@Injectable()
export class LogRedactionService {
    redact<T>(value: T): T {
        return this.redactValue(value, new WeakSet<object>()) as T;
    }

    private redactValue(value: unknown, seen: WeakSet<object>): unknown {
        if (value === null || value === undefined) {
            return value;
        }

        if (typeof value !== 'object') {
            return value;
        }

        if (value instanceof Error) {
            return value;
        }

        if (seen.has(value)) {
            return '[Circular]';
        }

        seen.add(value);

        if (Array.isArray(value)) {
            return value.map(item => this.redactValue(item, seen));
        }

        const redacted: Record<string, unknown> = {};

        for (const [key, nestedValue] of Object.entries(value)) {
            if (LOGGER_REDACTED_KEYS.has(normalizeRedactionKey(key))) {
                redacted[key] = LOG_REDACTION_CENSOR;
            } else {
                redacted[key] = this.redactValue(nestedValue, seen);
            }
        }

        return redacted;
    }
}

function normalizeRedactionKey(key: string): string {
    return key.toLowerCase().replace(/[_-]/g, '');
}
