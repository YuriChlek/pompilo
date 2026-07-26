import { Inject } from '@nestjs/common';
import type { LoggerChannel } from '@/module-logger/interfaces/logger.interfaces';
import { getLoggerChannelToken } from '@/module-logger/tokens/logger.tokens';

export function InjectLogger(channel: LoggerChannel): ReturnType<typeof Inject> {
    return Inject(getLoggerChannelToken(channel));
}
