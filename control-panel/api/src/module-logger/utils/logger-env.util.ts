import { loadApiEnvFiles } from '@config/api-env.config';
import { validateLoggerEnvironment } from '@/config/environment.validation';
import type { LoggerConfig, RuntimeLogger } from '@/module-logger/interfaces/logger.interfaces';
import { RuntimeLoggerFactoryService } from '@/module-logger/services/runtime-logger-factory.service';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';

export function createAppLoggerConfigFromEnv(): LoggerConfig {
    loadApiEnvFiles();

    const loggerEnvironment = validateLoggerEnvironment(process.env);

    return {
        environment: loggerEnvironment.NODE_ENV,
        format: loggerEnvironment.LOG_FORMAT,
        level: loggerEnvironment.LOG_LEVEL,
        output: loggerEnvironment.LOG_OUTPUT,
        fileDir: loggerEnvironment.LOG_FILE_DIR,
        fileMaxDays: loggerEnvironment.LOG_FILE_MAX_DAYS,
        redactionEnabled: loggerEnvironment.LOG_REDACTION_ENABLED,
        structuredBootstrapEnabled: loggerEnvironment.LOG_STRUCTURED_BOOTSTRAP_ENABLED,
        requestIdHeader: loggerEnvironment.LOG_REQUEST_ID_HEADER,
        serviceName: loggerEnvironment.SERVICE_NAME,
    };
}

export function createBootstrapLoggerConfigFromEnv(): LoggerConfig {
    return createAppLoggerConfigFromEnv();
}

export function createRuntimeLoggerFactoryFromEnv(
    config = createBootstrapLoggerConfigFromEnv(),
): RuntimeLoggerFactoryService {
    return new RuntimeLoggerFactoryService(config, new LogRedactionService());
}

export function createStructuredBootstrapLoggerFromEnv(
    config = createBootstrapLoggerConfigFromEnv(),
): RuntimeLogger | undefined {
    return createRuntimeLoggerFactoryFromEnv(config).createStructuredBootstrapLogger();
}
