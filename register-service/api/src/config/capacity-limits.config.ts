import { ConfigService } from '@nestjs/config';

export const INITIAL_CAPACITY_LIMITS = {
    criticalEmailRequestsPerSecond: 100,
    emailDeliveriesPerMinute: 6000,
    mailQueueLimiterDurationMs: 1000,
    mailWorkerConcurrencyPerApiReplica: 4,
    dbPoolMaxPerApiReplica: 30,
    redisMaxMemory: '2gb',
    redisEvictionPolicy: 'noeviction',
} as const;

export const getDbPoolMax = (configService: ConfigService): number => {
    const configuredMax = Number(configService.get<string>('DB_POOL_MAX'));

    if (!Number.isInteger(configuredMax) || configuredMax <= 0) {
        return INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica;
    }

    return Math.min(configuredMax, INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica);
};
