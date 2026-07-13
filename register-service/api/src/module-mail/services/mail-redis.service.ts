import { Injectable } from '@nestjs/common';
import { RedisService } from '@/common/redis/redis.service';

@Injectable()
export class MailRedisService {
    private readonly FAILURE_COUNT_KEY = 'mail:failure_count';
    private readonly DELIVERY_ERRORS_KEY = 'mail:delivery_errors';
    private readonly LAST_SUCCESSFUL_SEND_KEY = 'mail:last_successful_send';
    private readonly READINESS_STATUS_KEY = 'mail:readiness_status';
    private readonly MAX_ERRORS = 10;

    constructor(private readonly redisService: RedisService) {}

    async setReadinessStatus(status: string, error?: string, ttlSeconds = 300): Promise<void> {
        const client = this.redisService.getClient();
        const data = JSON.stringify({
            status,
            error: error || null,
            timestamp: new Date().toISOString(),
        });
        await client.set(this.READINESS_STATUS_KEY, data, 'EX', ttlSeconds);
    }

    async getReadinessStatus(): Promise<{
        status: string;
        error: string | null;
        timestamp: string;
    } | null> {
        const client = this.redisService.getClient();
        const data = await client.get(this.READINESS_STATUS_KEY);
        if (!data) return null;
        try {
            return JSON.parse(data) as {
                status: string;
                error: string | null;
                timestamp: string;
            };
        } catch {
            return null;
        }
    }

    async incrementFailureCount(): Promise<number> {
        const client = this.redisService.getClient();
        const count = await client.incr(this.FAILURE_COUNT_KEY);
        // Set TTL (30 minutes) for failure counter so it doesn't stay stale forever
        await client.expire(this.FAILURE_COUNT_KEY, 1800);
        return count;
    }

    async resetFailureCount(): Promise<void> {
        const client = this.redisService.getClient();
        await client.del(this.FAILURE_COUNT_KEY);
    }

    async getFailureCount(): Promise<number> {
        const client = this.redisService.getClient();
        const count = await client.get(this.FAILURE_COUNT_KEY);
        return count ? parseInt(count, 10) : 0;
    }

    async logDeliveryError(code: string, message: string): Promise<void> {
        const client = this.redisService.getClient();
        const error = JSON.stringify({
            code,
            message,
            timestamp: new Date().toISOString(),
        });

        await client.lpush(this.DELIVERY_ERRORS_KEY, error);
        await client.ltrim(this.DELIVERY_ERRORS_KEY, 0, this.MAX_ERRORS - 1);
        // Set TTL for errors list as well
        await client.expire(this.DELIVERY_ERRORS_KEY, 3600 * 24); // 24 hours
    }

    async getDeliveryErrors(): Promise<
        Array<{ code: string; message: string; timestamp: string }>
    > {
        const client = this.redisService.getClient();
        const errors = await client.lrange(this.DELIVERY_ERRORS_KEY, 0, -1);
        return errors.map(
            e => JSON.parse(e) as { code: string; message: string; timestamp: string },
        );
    }

    async logSuccessfulDelivery(): Promise<void> {
        const client = this.redisService.getClient();
        await client.set(this.LAST_SUCCESSFUL_SEND_KEY, new Date().toISOString());
        await this.resetFailureCount(); // Reset failures on success
        await client.del(this.DELIVERY_ERRORS_KEY); // Also clear errors on success
    }

    async getLastSuccessfulSendAt(): Promise<string | null> {
        const client = this.redisService.getClient();
        return await client.get(this.LAST_SUCCESSFUL_SEND_KEY);
    }

    private readonly METRICS_SUCCESS_KEY = 'mail:metrics:success_total';
    private readonly METRICS_ERRORS_KEY_PREFIX = 'mail:metrics:errors_total:';

    async incrementSuccessCount(): Promise<void> {
        const client = this.redisService.getClient();
        await client.incr(this.METRICS_SUCCESS_KEY);
    }

    async incrementErrorCount(
        category: 'network' | 'auth' | 'rate-limit' | 'rejection',
    ): Promise<void> {
        const client = this.redisService.getClient();
        await client.incr(`${this.METRICS_ERRORS_KEY_PREFIX}${category}`);
    }

    async getSuccessCount(): Promise<number> {
        const client = this.redisService.getClient();
        const val = await client.get(this.METRICS_SUCCESS_KEY);
        return val ? parseInt(val, 10) : 0;
    }

    async getErrorCounts(): Promise<
        Record<'network' | 'auth' | 'rate-limit' | 'rejection', number>
    > {
        const client = this.redisService.getClient();
        const keys = [
            `${this.METRICS_ERRORS_KEY_PREFIX}network`,
            `${this.METRICS_ERRORS_KEY_PREFIX}auth`,
            `${this.METRICS_ERRORS_KEY_PREFIX}rate-limit`,
            `${this.METRICS_ERRORS_KEY_PREFIX}rejection`,
        ];
        const vals = await client.mget(...keys);
        return {
            network: vals[0] ? parseInt(vals[0], 10) : 0,
            auth: vals[1] ? parseInt(vals[1], 10) : 0,
            'rate-limit': vals[2] ? parseInt(vals[2], 10) : 0,
            rejection: vals[3] ? parseInt(vals[3], 10) : 0,
        };
    }
}
