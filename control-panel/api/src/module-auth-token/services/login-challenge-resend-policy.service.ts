import { Injectable } from '@nestjs/common';
import { createHash } from 'crypto';
import { RedisService } from '@/common/redis/redis.service';
import {
    LOGIN_CHALLENGE_RESEND_COOLDOWN_SECONDS,
    LOGIN_CHALLENGE_RESEND_IP_MAX_ATTEMPTS,
    LOGIN_CHALLENGE_RESEND_MAX_ATTEMPTS,
    LOGIN_CHALLENGE_RESEND_WINDOW_SECONDS,
} from '@/module-auth-token/constants/login-challenge.constants';
import type {
    LoginChallengeResendPolicyInput,
    LoginChallengeResendPolicyLimitReason,
    LoginChallengeResendPolicyResult,
} from '@/module-auth-token/types/login-challenge.types';

const INCREMENT_WITH_TTL_SCRIPT = `
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return current
`;

@Injectable()
export class LoginChallengeResendPolicyService {
    private readonly redis: ReturnType<RedisService['getClient']>;

    constructor(private readonly redisService: RedisService) {
        this.redis = redisService.getClient();
    }

    async reserveResendAttempt(
        input: LoginChallengeResendPolicyInput,
    ): Promise<LoginChallengeResendPolicyResult> {
        try {
            const cooldownRetryAfterSeconds = await this.getRetryAfterSeconds(
                this.buildCooldownKey(input),
            );

            if (cooldownRetryAfterSeconds > 0) {
                this.redisService.recordSuccess();
                return {
                    allowed: false,
                    reason: 'cooldown',
                    retryAfterSeconds: cooldownRetryAfterSeconds,
                };
            }

            const userDeviceLimit = await this.incrementBucket(
                this.buildUserDeviceWindowKey(input),
                LOGIN_CHALLENGE_RESEND_MAX_ATTEMPTS,
                'user_device_rate_limited',
            );

            if (userDeviceLimit) {
                this.redisService.recordSuccess();
                return userDeviceLimit;
            }

            const ipLimit = await this.incrementBucket(
                this.buildIpWindowKey(input),
                LOGIN_CHALLENGE_RESEND_IP_MAX_ATTEMPTS,
                'ip_rate_limited',
            );

            if (ipLimit) {
                this.redisService.recordSuccess();
                return ipLimit;
            }

            await this.redis.set(
                this.buildCooldownKey(input),
                '1',
                'PX',
                LOGIN_CHALLENGE_RESEND_COOLDOWN_SECONDS * 1000,
            );
            this.redisService.recordSuccess();

            return {
                allowed: true,
                retryAfterSeconds: 0,
            };
        } catch (error) {
            this.redisService.recordFailure();
            throw error;
        }
    }

    private async incrementBucket(
        key: string,
        maxAttempts: number,
        reason: LoginChallengeResendPolicyLimitReason,
    ): Promise<LoginChallengeResendPolicyResult | null> {
        const current = Number(
            await this.redis.eval(
                INCREMENT_WITH_TTL_SCRIPT,
                1,
                key,
                LOGIN_CHALLENGE_RESEND_WINDOW_SECONDS * 1000,
            ),
        );

        if (current <= maxAttempts) {
            return null;
        }

        return {
            allowed: false,
            reason,
            retryAfterSeconds: await this.getRetryAfterSeconds(key),
        };
    }

    private async getRetryAfterSeconds(key: string): Promise<number> {
        const ttlMs = await this.redis.pttl(key);

        if (ttlMs <= 0) {
            return 0;
        }

        return Math.ceil(ttlMs / 1000);
    }

    private buildCooldownKey(input: LoginChallengeResendPolicyInput): string {
        return `login-challenge:resend:cooldown:${this.hashValue(
            `${input.userId}:${input.realm}:${input.deviceId}`,
        )}`;
    }

    private buildUserDeviceWindowKey(input: LoginChallengeResendPolicyInput): string {
        return `login-challenge:resend:user-device:${this.hashValue(
            `${input.userId}:${input.realm}:${input.deviceId}`,
        )}`;
    }

    private buildIpWindowKey(input: LoginChallengeResendPolicyInput): string {
        return `login-challenge:resend:ip:${this.hashValue(input.ipAddress)}`;
    }

    private hashValue(value: string): string {
        return createHash('sha256').update(value).digest('hex');
    }
}
