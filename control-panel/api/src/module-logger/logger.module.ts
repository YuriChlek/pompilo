import { Global, Module, type Provider } from '@nestjs/common';
import { getLoggerChannelToken, LOGGER_CONFIG } from '@/module-logger/tokens/logger.tokens';
import { createAppLoggerConfigFromEnv } from '@/module-logger/utils/logger-env.util';
import { LogRedactionService } from '@/module-logger/services/log-redaction.service';
import { LoggerFactoryService } from '@/module-logger/services/logger-factory.service';
import type { LoggerConfig } from '@/module-logger/interfaces/logger.interfaces';
import { LOGGER_CHANNELS } from '@/module-logger/constants/logger.constants';
import { RuntimeLoggerFactoryService } from '@/module-logger/services/runtime-logger-factory.service';

const loggerChannelProviders: Provider[] = LOGGER_CHANNELS.map(channel => ({
    provide: getLoggerChannelToken(channel),
    inject: [LoggerFactoryService],
    useFactory: (loggerFactory: LoggerFactoryService) => loggerFactory.createChannelLogger(channel),
}));

const loggerChannelProviderTokens = LOGGER_CHANNELS.map(channel => getLoggerChannelToken(channel));

@Global()
@Module({
    providers: [
        {
            provide: LOGGER_CONFIG,
            useFactory: createAppLoggerConfigFromEnv,
        },
        LogRedactionService,
        {
            provide: RuntimeLoggerFactoryService,
            inject: [LOGGER_CONFIG, LogRedactionService],
            useFactory: (config: LoggerConfig, redactor: LogRedactionService) =>
                new RuntimeLoggerFactoryService(config, redactor),
        },
        {
            provide: LoggerFactoryService,
            inject: [RuntimeLoggerFactoryService],
            useFactory: (runtimeLoggerFactory: RuntimeLoggerFactoryService) =>
                new LoggerFactoryService(runtimeLoggerFactory),
        },
        ...loggerChannelProviders,
    ],
    exports: [
        LOGGER_CONFIG,
        RuntimeLoggerFactoryService,
        LoggerFactoryService,
        ...loggerChannelProviderTokens,
    ],
})
export class LoggerModule {}
