import { ConfigService } from '@nestjs/config';
import { QueueOptions } from 'bullmq';

export const getBullMqConfig = (configService: ConfigService): QueueOptions => {
    return {
        connection: {
            host: configService.getOrThrow<string>('REDIS_HOST', 'localhost'),
            port: Number(configService.getOrThrow<string>('REDIS_PORT', '6379')),
        },
        defaultJobOptions: {
            attempts: 5,
            backoff: {
                type: 'exponential',
                delay: 1000,
            },
            removeOnComplete: true,
        },
    };
};
