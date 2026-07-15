import { Injectable } from '@nestjs/common';
import { createHash } from 'crypto';
import { RedisService } from '@/common/redis/redis.service';
import { EMAIL_FLOW_RATE_LIMIT_CONFIG } from '@/common/rate-limiting/constants/email-flow-rate-limit.constants';
import type {
    EmailFlowRateLimitBucketConfig,
    EmailFlowRateLimitCheckInput,
    EmailFlowRateLimitCheckResult,
    EmailRateLimitDimension,
} from '@/common/rate-limiting/interfaces/email-flow-rate-limit.interfaces';

const INCREMENT_WITH_TTL_SCRIPT = `
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return current
`;

@Injectable()
export class EmailFlowRateLimitService {
    private readonly redis: ReturnType<RedisService['getClient']>;

    constructor(redisService: RedisService) {
        this.redis = redisService.getClient();
    }

    async check(input: EmailFlowRateLimitCheckInput): Promise<EmailFlowRateLimitCheckResult> {
        const config = EMAIL_FLOW_RATE_LIMIT_CONFIG[input.flow];
        const checks = [
            this.checkBucket(this.buildKey(input.flow, 'ip', input.ipAddress), config.ip),
        ];

        const recipientEmail = this.normalizeRecipient(input.recipientEmail);
        if (recipientEmail) {
            checks.push(
                this.checkBucket(
                    this.buildKey(input.flow, 'recipient', recipientEmail),
                    config.recipient,
                ),
            );
        }

        const results = await Promise.all(checks);
        const limitedResults = results.filter(result => result.limited);

        return {
            limited: limitedResults.length > 0,
            retryAfterSeconds:
                limitedResults.length > 0
                    ? Math.max(...limitedResults.map(result => result.retryAfterSeconds))
                    : 0,
        };
    }

    private async checkBucket(
        key: string,
        config: EmailFlowRateLimitBucketConfig,
    ): Promise<EmailFlowRateLimitCheckResult> {
        const current = Number(
            await this.redis.eval(INCREMENT_WITH_TTL_SCRIPT, 1, key, config.windowSeconds * 1000),
        );

        if (current <= config.maxRequests) {
            return { limited: false, retryAfterSeconds: 0 };
        }

        const ttlMs = await this.redis.pttl(key);
        const retryAfterSeconds = ttlMs > 0 ? Math.ceil(ttlMs / 1000) : config.windowSeconds;

        return { limited: true, retryAfterSeconds };
    }

    private buildKey(
        flow: EmailFlowRateLimitCheckInput['flow'],
        dimension: EmailRateLimitDimension,
        value: string,
    ): string {
        return `rate-limit:email-flow:${flow}:${dimension}:${this.hashValue(value)}`;
    }

    private hashValue(value: string): string {
        return createHash('sha256').update(value).digest('hex');
    }

    private normalizeRecipient(email: string | undefined): string | undefined {
        const normalized = email?.trim().toLowerCase();
        return normalized && normalized.length > 0 ? normalized : undefined;
    }
}
