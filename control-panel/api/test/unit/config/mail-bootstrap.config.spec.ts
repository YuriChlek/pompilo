import { ConfigService } from '@nestjs/config';
import { getMailBootstrapConfig } from '@/config/mail-bootstrap.config';

const mailBootstrapConfig = () => getMailBootstrapConfig(new ConfigService(process.env));

describe('mailBootstrapConfig', () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        jest.resetModules();
        process.env = { ...originalEnv };
    });

    afterAll(() => {
        process.env = originalEnv;
    });

    it('should parse environment variables correctly', () => {
        process.env.MAIL_SETTINGS_ENCRYPTION_KEY = 'secret-key';
        process.env.MAIL_QUEUE_LIMITER_MAX = '50';
        process.env.MAIL_QUEUE_LIMITER_DURATION = '2000';
        process.env.MAIL_WORKER_CONCURRENCY = '8';
        process.env.MAIL_RETRY_ATTEMPTS = '5';
        process.env.MAIL_RETRY_BACKOFF_DELAY = '3000';
        process.env.MAIL_RETRY_BACKOFF_TYPE = 'exponential';
        process.env.MAIL_RETRY_BACKOFF_JITTER = '0.35';

        const config = mailBootstrapConfig();

        expect(config).toEqual({
            encryptionKey: 'secret-key',
            queueLimiterMax: 50,
            queueLimiterDuration: 2000,
            workerConcurrency: 8,
            retryAttempts: 5,
            retryBackoffDelay: 3000,
            retryBackoffType: 'exponential',
            retryBackoffJitter: 0.35,
        });
    });

    it('should use default values if environment variables are missing', () => {
        delete process.env.MAIL_SETTINGS_ENCRYPTION_KEY;
        delete process.env.MAIL_QUEUE_LIMITER_MAX;
        delete process.env.MAIL_QUEUE_LIMITER_DURATION;
        delete process.env.MAIL_WORKER_CONCURRENCY;
        delete process.env.MAIL_RETRY_ATTEMPTS;
        delete process.env.MAIL_RETRY_BACKOFF_DELAY;
        delete process.env.MAIL_RETRY_BACKOFF_TYPE;
        delete process.env.MAIL_RETRY_BACKOFF_JITTER;

        const config = mailBootstrapConfig();

        expect(config.encryptionKey).toBeUndefined();
        expect(config.queueLimiterMax).toBe(100);
        expect(config.queueLimiterDuration).toBe(1000);
        expect(config.workerConcurrency).toBe(4);
        expect(config.retryAttempts).toBe(3);
        expect(config.retryBackoffDelay).toBe(2000);
        expect(config.retryBackoffType).toBe('exponential');
        expect(config.retryBackoffJitter).toBe(0.2);
    });

    it('should clamp retry backoff jitter to BullMQ supported ratio range', () => {
        process.env.MAIL_RETRY_BACKOFF_JITTER = '1.5';
        expect(mailBootstrapConfig().retryBackoffJitter).toBe(1);

        process.env.MAIL_RETRY_BACKOFF_JITTER = '-0.5';
        expect(mailBootstrapConfig().retryBackoffJitter).toBe(0);
    });

    it('should fall back for invalid worker and retry limits', () => {
        process.env.MAIL_QUEUE_LIMITER_MAX = '0';
        process.env.MAIL_QUEUE_LIMITER_DURATION = '-1';
        process.env.MAIL_WORKER_CONCURRENCY = 'not-a-number';
        process.env.MAIL_RETRY_ATTEMPTS = '2.5';
        process.env.MAIL_RETRY_BACKOFF_DELAY = '0';

        const config = mailBootstrapConfig();

        expect(config.queueLimiterMax).toBe(100);
        expect(config.queueLimiterDuration).toBe(1000);
        expect(config.workerConcurrency).toBe(4);
        expect(config.retryAttempts).toBe(3);
        expect(config.retryBackoffDelay).toBe(2000);
    });
});
