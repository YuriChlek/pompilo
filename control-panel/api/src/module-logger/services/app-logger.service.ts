import { Injectable, type LoggerService, type LogLevel } from '@nestjs/common';
import pino, { type Logger as PinoLogger, type LoggerOptions } from 'pino';
import {
    LOG_REDACTION_CENSOR,
    PINO_REDACT_PATHS,
} from '@/module-logger/constants/logger.constants';
import type {
    LoggerConfig,
    LoggerLevel,
    NormalizedLogPayload,
    SerializedError,
} from '@/module-logger/interfaces/logger.interfaces';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';
import { getRequestLogContext } from '@/module-logger/utils/request-log-context.util';
import { createRotatingLogFileStream } from '@/module-logger/utils/rotating-log-file-stream.util';

type LogMethod = 'fatal' | 'error' | 'warn' | 'info' | 'debug' | 'trace';

@Injectable()
export class AppLoggerService implements LoggerService {
    private readonly logger: PinoLogger;
    private readonly redactor: LogRedactionService;

    constructor(
        private readonly config: LoggerConfig,
        logger?: PinoLogger,
        redactor = new LogRedactionService(),
    ) {
        this.redactor = redactor;
        this.logger = logger ?? createPinoLogger(config);
    }

    log(message: unknown, ...optionalParams: unknown[]): void {
        this.write('info', message, optionalParams);
    }

    error(message: unknown, ...optionalParams: unknown[]): void {
        this.write('error', message, optionalParams);
    }

    warn(message: unknown, ...optionalParams: unknown[]): void {
        this.write('warn', message, optionalParams);
    }

    debug(message: unknown, ...optionalParams: unknown[]): void {
        this.write('debug', message, optionalParams);
    }

    verbose(message: unknown, ...optionalParams: unknown[]): void {
        this.write('trace', message, optionalParams);
    }

    fatal(message: unknown, ...optionalParams: unknown[]): void {
        this.write('fatal', message, optionalParams);
    }

    setLogLevels?(levels: LogLevel[]): void {
        if (levels.length === 0) {
            this.logger.level = 'silent';
            return;
        }

        const mappedLevels = levels.map(mapNestLogLevel);
        const orderedLevels: LoggerLevel[] = ['trace', 'debug', 'info', 'warn', 'error', 'fatal'];
        const lowestLevel = orderedLevels.find(level => mappedLevels.includes(level));
        this.logger.level = lowestLevel ?? 'silent';
    }

    private write(level: LogMethod, message: unknown, optionalParams: unknown[]): void {
        if (this.logger.level === 'silent') {
            return;
        }

        const { messageText, payload } = this.normalizeLogArguments(message, optionalParams);
        const payloadWithRequestContext = this.attachRequestContext(payload);
        const redactedPayload = this.config.redactionEnabled
            ? this.redactor.redact(payloadWithRequestContext)
            : payloadWithRequestContext;

        this.logger[level](redactedPayload, messageText);
    }

    private attachRequestContext(payload: NormalizedLogPayload): NormalizedLogPayload {
        const requestLogContext = getRequestLogContext();

        if (!requestLogContext?.requestId || payload.requestId) {
            return payload;
        }

        return {
            requestId: requestLogContext.requestId,
            ...payload,
        };
    }

    private normalizeLogArguments(
        message: unknown,
        optionalParams: unknown[],
    ): { messageText: string; payload: NormalizedLogPayload } {
        const params = [...optionalParams];
        const payload: NormalizedLogPayload = {};

        if (params.length > 0 && typeof params[params.length - 1] === 'string') {
            payload.context = params.pop() as string;
        }

        if (message instanceof Error) {
            payload.error = serializeError(message);
            return {
                messageText: message.message,
                payload: this.attachMetadata(payload, params),
            };
        }

        const errorParamIndex = params.findIndex(param => param instanceof Error);
        if (errorParamIndex >= 0) {
            const [error] = params.splice(errorParamIndex, 1);
            payload.error = serializeError(error as Error);
        }

        if (params.length > 0 && typeof params[0] === 'string' && payload.context) {
            payload.stack = params.shift() as string;
        }

        return {
            messageText: stringifyLogMessage(message),
            payload: this.attachMetadata(payload, params),
        };
    }

    private attachMetadata(
        payload: NormalizedLogPayload,
        metadataParams: unknown[],
    ): NormalizedLogPayload {
        if (metadataParams.length === 0) {
            return payload;
        }

        return {
            ...payload,
            metadata: metadataParams.length === 1 ? metadataParams[0] : metadataParams,
        };
    }
}

export function createPinoLogger(config: LoggerConfig): PinoLogger {
    const options: LoggerOptions = {
        level: config.level,
        messageKey: 'message',
        base: {
            service: config.serviceName,
            environment: config.environment,
        },
        formatters: {
            level(label) {
                return { level: label };
            },
        },
        timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
        redact: config.redactionEnabled
            ? {
                  paths: [...PINO_REDACT_PATHS],
                  censor: LOG_REDACTION_CENSOR,
              }
            : undefined,
    };

    if (config.output === 'file') {
        return pino(
            options,
            createRotatingLogFileStream({
                directory: config.fileDir,
                maxDays: config.fileMaxDays,
                serviceName: config.serviceName,
            }),
        );
    }

    if (config.format === 'pretty') {
        return pino({
            ...options,
            transport: {
                target: 'pino-pretty',
                options: {
                    colorize: config.environment === 'development',
                    ignore: 'pid,hostname',
                    messageKey: 'message',
                    singleLine: true,
                    translateTime: 'yyyy-mm-dd HH:MM:ss',
                },
            },
        });
    }

    return pino(options);
}

function stringifyLogMessage(message: unknown): string {
    if (typeof message === 'string') {
        return message;
    }

    if (
        typeof message === 'number' ||
        typeof message === 'boolean' ||
        typeof message === 'bigint'
    ) {
        return String(message);
    }

    if (message === null || message === undefined) {
        return String(message);
    }

    try {
        return JSON.stringify(message);
    } catch {
        return '[Unserializable log message]';
    }
}

function serializeError(error: Error): SerializedError {
    return {
        name: error.name,
        message: error.message,
        stack: error.stack,
        cause:
            error.cause instanceof Error
                ? serializeError(error.cause)
                : error.cause === undefined
                  ? undefined
                  : error.cause,
    };
}

function mapNestLogLevel(level: LogLevel): LoggerLevel {
    if (level === 'log') {
        return 'info';
    }

    if (level === 'verbose') {
        return 'trace';
    }

    return level;
}
