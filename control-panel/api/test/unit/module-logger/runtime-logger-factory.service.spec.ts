import type { Logger as PinoLogger } from 'pino';
import { RuntimeLoggerFactoryService } from '@/module-logger/services/runtime-logger-factory.service';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';
import type { LoggerConfig } from '@/module-logger/interfaces/logger.interfaces';

describe('RuntimeLoggerFactoryService', () => {
    const productionConfig: LoggerConfig = {
        environment: 'production',
        format: 'json',
        level: 'info',
        output: 'console',
        fileDir: 'logs',
        fileMaxDays: 14,
        redactionEnabled: true,
        structuredBootstrapEnabled: true,
        requestIdHeader: 'x-request-id',
        serviceName: 'pampilo-api',
    };
    const developmentConfig: LoggerConfig = {
        ...productionConfig,
        environment: 'development',
        level: 'debug',
        structuredBootstrapEnabled: false,
    };

    function createPinoMock() {
        return {
            level: 'info',
            fatal: jest.fn(),
            error: jest.fn(),
            warn: jest.fn(),
            info: jest.fn(),
            debug: jest.fn(),
            trace: jest.fn(),
            child: jest.fn(),
        } as unknown as PinoLogger;
    }

    it('creates a Pino-backed bootstrap logger in production', () => {
        const pinoLogger = createPinoMock();
        const factory = new RuntimeLoggerFactoryService(
            productionConfig,
            new LogRedactionService(),
            pinoLogger,
        );

        const logger = factory.createStructuredBootstrapLogger();

        expect(logger).toBeDefined();
        if (!logger) {
            throw new Error('Expected production bootstrap logger');
        }
        logger.log('Production boot', 'Bootstrap');

        expect(pinoLogger.info).toHaveBeenCalledWith(
            {
                context: 'Bootstrap',
            },
            'Production boot',
        );
    });

    it('does not create a bootstrap logger in development by default', () => {
        const factory = new RuntimeLoggerFactoryService(
            developmentConfig,
            new LogRedactionService(),
        );

        const logger = factory.createStructuredBootstrapLogger();

        expect(logger).toBeUndefined();
    });

    it('creates a Pino-backed bootstrap logger in development when temporarily enabled', () => {
        const pinoLogger = createPinoMock();
        const factory = new RuntimeLoggerFactoryService(
            {
                ...developmentConfig,
                structuredBootstrapEnabled: true,
            },
            new LogRedactionService(),
            pinoLogger,
        );

        const logger = factory.createStructuredBootstrapLogger();

        expect(logger).toBeDefined();
        if (!logger) {
            throw new Error('Expected temporary structured bootstrap logger');
        }
        logger.log('Temporary structured boot', 'Bootstrap');

        expect(pinoLogger.info).toHaveBeenCalledWith(
            {
                context: 'Bootstrap',
            },
            'Temporary structured boot',
        );
    });
});
