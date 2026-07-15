import { Logger, type LoggerService, type LogLevel } from '@nestjs/common';

export class NestLoggerAdapter implements LoggerService {
    private readonly logger: Logger;

    constructor(context = 'NestLogger') {
        this.logger = new Logger(context);
    }

    log(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.log(message, ...optionalParams);
    }

    error(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.error(message, ...optionalParams);
    }

    warn(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.warn(message, ...optionalParams);
    }

    debug(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.debug(message, ...optionalParams);
    }

    verbose(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.verbose(message, ...optionalParams);
    }

    fatal(message: unknown, ...optionalParams: unknown[]): void {
        this.logger.fatal(message, ...optionalParams);
    }

    setLogLevels(levels: LogLevel[]): void {
        Logger.overrideLogger(levels);
    }
}
