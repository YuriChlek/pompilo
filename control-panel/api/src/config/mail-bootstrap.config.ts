import { ConfigService } from '@nestjs/config';
import { INITIAL_CAPACITY_LIMITS } from '@/config/capacity-limits.config';

export interface MailBootstrapConfig {
    encryptionKey?: string;
    queueLimiterMax: number;
    queueLimiterDuration: number;
    workerConcurrency: number;
    retryAttempts: number;
    retryBackoffDelay: number;
    retryBackoffType: string;
    retryBackoffJitter: number;
}

export const getMailBootstrapConfig = (configService: ConfigService): MailBootstrapConfig => {
    // 1. Compatibility check: if configService returns the whole namespace config, use it directly.
    // This is particularly important for unit tests that mock configService.get() to return the configuration object.
    const nsConfig = configService.get<MailBootstrapConfig>('mail-bootstrap');
    if (
        nsConfig &&
        typeof nsConfig === 'object' &&
        ('retryAttempts' in nsConfig || 'queueLimiterMax' in nsConfig)
    ) {
        return nsConfig;
    }

    const parseOptionalNumber = (key: string, fallback: number): number => {
        const val = configService.get<string>(key);
        if (val === undefined) {
            return fallback;
        }

        const parsed = Number(val);
        return Number.isFinite(parsed) ? parsed : fallback;
    };

    const parsePositiveInteger = (key: string, fallback: number): number => {
        const parsed = parseOptionalNumber(key, fallback);
        return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
    };

    const parseRetryBackoffJitter = (): number => {
        const jitter = parseOptionalNumber('MAIL_RETRY_BACKOFF_JITTER', 0.2);
        return Math.min(Math.max(jitter, 0), 1);
    };

    return {
        encryptionKey: configService.get<string>('MAIL_SETTINGS_ENCRYPTION_KEY'),
        queueLimiterMax: parsePositiveInteger(
            'MAIL_QUEUE_LIMITER_MAX',
            INITIAL_CAPACITY_LIMITS.criticalEmailRequestsPerSecond,
        ),
        queueLimiterDuration: parsePositiveInteger(
            'MAIL_QUEUE_LIMITER_DURATION',
            INITIAL_CAPACITY_LIMITS.mailQueueLimiterDurationMs,
        ),
        workerConcurrency: parsePositiveInteger(
            'MAIL_WORKER_CONCURRENCY',
            INITIAL_CAPACITY_LIMITS.mailWorkerConcurrencyPerApiReplica,
        ),
        retryAttempts: parsePositiveInteger('MAIL_RETRY_ATTEMPTS', 3),
        retryBackoffDelay: parsePositiveInteger('MAIL_RETRY_BACKOFF_DELAY', 2000),
        retryBackoffType: configService.get<string>('MAIL_RETRY_BACKOFF_TYPE') || 'exponential',
        retryBackoffJitter: parseRetryBackoffJitter(),
    };
};
