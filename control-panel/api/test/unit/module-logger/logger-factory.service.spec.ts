import { Logger } from '@nestjs/common';
import type { Logger as PinoLogger } from 'pino';
import { LoggerFactoryService } from '@/module-logger/services/logger-factory.service';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';
import type { LoggerConfig } from '@/module-logger/interfaces/logger.interfaces';
import { RuntimeLoggerFactoryService } from '@/module-logger/services/runtime-logger-factory.service';

describe('LoggerFactoryService', () => {
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
        const childLogger = {
            level: 'debug',
            fatal: jest.fn(),
            error: jest.fn(),
            warn: jest.fn(),
            info: jest.fn(),
            debug: jest.fn(),
            trace: jest.fn(),
            child: jest.fn(),
        } as unknown as PinoLogger;
        const rootLogger = {
            level: 'debug',
            child: jest.fn(() => childLogger),
        } as unknown as PinoLogger & {
            child: jest.Mock;
        };

        return { childLogger, rootLogger };
    }

    it('creates production channel loggers from the single Pino root logger child API', () => {
        const { childLogger, rootLogger } = createPinoMock();
        const runtimeFactory = new RuntimeLoggerFactoryService(
            productionConfig,
            new LogRedactionService(),
            rootLogger,
        );
        const factory = new LoggerFactoryService(runtimeFactory);

        const logger = factory.createChannelLogger('mail');

        logger.log('Mail channel ready', 'MailService');

        expect(rootLogger.child).toHaveBeenCalledWith({ channel: 'mail' });
        expect(childLogger.info).toHaveBeenCalledWith(
            {
                context: 'MailService',
            },
            'Mail channel ready',
        );
    });

    it('caches production channel loggers so repeated injections do not create new child loggers', () => {
        const { rootLogger } = createPinoMock();
        const runtimeFactory = new RuntimeLoggerFactoryService(
            productionConfig,
            new LogRedactionService(),
            rootLogger,
        );
        const factory = new LoggerFactoryService(runtimeFactory);

        const firstLogger = factory.createChannelLogger('chat');
        const secondLogger = factory.createChannelLogger('chat');

        expect(secondLogger).toBe(firstLogger);
        expect(rootLogger.child).toHaveBeenCalledTimes(1);
        expect(rootLogger.child).toHaveBeenCalledWith({ channel: 'chat' });
    });

    it('creates non-production channel loggers as adapters over the standard Nest Logger', () => {
        const loggerSpy = jest.spyOn(Logger.prototype, 'log').mockImplementation(() => {});
        const runtimeFactory = new RuntimeLoggerFactoryService(
            developmentConfig,
            new LogRedactionService(),
        );
        const factory = new LoggerFactoryService(runtimeFactory);

        const logger = factory.createChannelLogger('mail');

        logger.log('Mail channel ready');

        expect(loggerSpy).toHaveBeenCalledWith('Mail channel ready');
        loggerSpy.mockRestore();
    });
});
