import type { Logger as PinoLogger } from 'pino';
import { AppLoggerService } from '@/module-logger/services/app-logger.service';
import type { LoggerConfig } from '@/module-logger/interfaces/logger.interfaces';
import { runWithRequestLogContext } from '@/module-logger/utils/request-log-context.util';

describe('AppLoggerService', () => {
    const config: LoggerConfig = {
        environment: 'test',
        format: 'json',
        level: 'debug',
        output: 'console',
        fileDir: 'logs',
        fileMaxDays: 14,
        redactionEnabled: true,
        structuredBootstrapEnabled: false,
        requestIdHeader: 'x-request-id',
        serviceName: 'pampilo-api',
    };

    function createPinoMock() {
        return {
            level: 'debug',
            fatal: jest.fn(),
            error: jest.fn(),
            warn: jest.fn(),
            info: jest.fn(),
            debug: jest.fn(),
            trace: jest.fn(),
            child: jest.fn(),
        } as unknown as PinoLogger;
    }

    it('normalizes Nest logger context and metadata without losing structured fields', () => {
        const pinoLogger = createPinoMock();
        const logger = new AppLoggerService(config, pinoLogger);

        logger.log('Mail job completed', { jobId: 'job-1' }, 'MailProcessorService');

        expect(pinoLogger.info).toHaveBeenCalledWith(
            {
                context: 'MailProcessorService',
                metadata: {
                    jobId: 'job-1',
                },
            },
            'Mail job completed',
        );
    });

    it('serializes Error objects from Nest-compatible error calls', () => {
        const pinoLogger = createPinoMock();
        const logger = new AppLoggerService(config, pinoLogger);
        const error = new Error('Database connection failed');

        logger.error('Failed to cleanup expired users', error, 'UserCleanupService');

        const errorMock = pinoLogger.error as unknown as jest.MockedFunction<
            (payload: unknown, message: string) => void
        >;
        const [payload, message] = errorMock.mock.calls[0] ?? [];
        const errorPayload = payload as {
            context?: string;
            error?: {
                name?: string;
                message?: string;
                stack?: unknown;
            };
        };

        expect(message).toBe('Failed to cleanup expired users');
        expect(errorPayload.context).toBe('UserCleanupService');
        expect(errorPayload.error).toEqual(
            expect.objectContaining({
                name: 'Error',
                message: 'Database connection failed',
            }),
        );
        expect(typeof errorPayload.error?.stack).toBe('string');
    });

    it('redacts sensitive nested metadata before writing logs', () => {
        const pinoLogger = createPinoMock();
        const logger = new AppLoggerService(config, pinoLogger);

        logger.warn(
            'Sensitive payload rejected',
            {
                password: 'plain-text',
                Authorization: 'Bearer secret',
                api_key: 'api-key',
                nested: {
                    refreshToken: 'refresh-token',
                    safe: 'value',
                },
            },
            'AuthService',
        );

        expect(pinoLogger.warn).toHaveBeenCalledWith(
            {
                context: 'AuthService',
                metadata: {
                    password: '[REDACTED]',
                    Authorization: '[REDACTED]',
                    api_key: '[REDACTED]',
                    nested: {
                        refreshToken: '[REDACTED]',
                        safe: 'value',
                    },
                },
            },
            'Sensitive payload rejected',
        );
    });

    it('adds requestId from async request context to log payloads', () => {
        const pinoLogger = createPinoMock();
        const logger = new AppLoggerService(config, pinoLogger);

        runWithRequestLogContext({ requestId: 'req-123' }, () => {
            logger.log('Request-scoped message', 'RequestScopedService');
        });

        expect(pinoLogger.info).toHaveBeenCalledWith(
            {
                requestId: 'req-123',
                context: 'RequestScopedService',
            },
            'Request-scoped message',
        );
    });

    it('maps Nest log levels to Pino levels', () => {
        const pinoLogger = createPinoMock();
        const logger = new AppLoggerService(config, pinoLogger);

        logger.setLogLevels?.(['error', 'warn']);

        expect(pinoLogger.level).toBe('warn');
    });
});
