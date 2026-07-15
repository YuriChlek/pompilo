import { Injectable } from '@nestjs/common';
import type { Logger as PinoLogger } from 'pino';
import { NestLoggerAdapter } from '@/module-logger/adapters/nest-logger.adapter';
import { AppLoggerService, createPinoLogger } from '@/module-logger/services/app-logger.service';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';
import type { LoggerConfig, RuntimeLogger } from '@/module-logger/interfaces/logger.interfaces';

@Injectable()
export class RuntimeLoggerFactoryService {
    private readonly rootPinoLogger?: PinoLogger;
    private readonly channelLoggers = new Map<string, RuntimeLogger>();

    constructor(
        private readonly config: LoggerConfig,
        private readonly redactor: LogRedactionService,
        rootPinoLogger?: PinoLogger,
    ) {
        this.rootPinoLogger =
            rootPinoLogger ??
            (this.shouldUseStructuredBootstrap() ? createPinoLogger(config) : undefined);
    }

    createStructuredBootstrapLogger(): RuntimeLogger | undefined {
        if (!this.shouldUseStructuredBootstrap()) {
            return undefined;
        }

        return new AppLoggerService(this.config, this.rootPinoLogger, this.redactor);
    }

    createChannelLogger(channel: string): RuntimeLogger {
        const existingLogger = this.channelLoggers.get(channel);

        if (existingLogger) {
            return existingLogger;
        }

        const logger = this.isProduction()
            ? new AppLoggerService(
                  this.config,
                  this.getRootPinoLogger().child({ channel }),
                  this.redactor,
              )
            : new NestLoggerAdapter(channel);

        this.channelLoggers.set(channel, logger);

        return logger;
    }

    private isProduction(): boolean {
        return this.config.environment === 'production';
    }

    private shouldUseStructuredBootstrap(): boolean {
        return this.isProduction() || this.config.structuredBootstrapEnabled;
    }

    private getRootPinoLogger(): PinoLogger {
        if (!this.rootPinoLogger) {
            throw new Error('Pino root logger is available only in production mode');
        }

        return this.rootPinoLogger;
    }
}
