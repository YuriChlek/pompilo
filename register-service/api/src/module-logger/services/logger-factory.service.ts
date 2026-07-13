import { Injectable } from '@nestjs/common';
import { RuntimeLoggerFactoryService } from '@/module-logger/services/runtime-logger-factory.service';
import type { RuntimeLogger } from '@/module-logger/interfaces/logger.interfaces';

@Injectable()
export class LoggerFactoryService {
    constructor(private readonly runtimeLoggerFactory: RuntimeLoggerFactoryService) {}

    createChannelLogger(channel: string): RuntimeLogger {
        return this.runtimeLoggerFactory.createChannelLogger(channel);
    }
}
