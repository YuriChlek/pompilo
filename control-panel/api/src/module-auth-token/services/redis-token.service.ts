import { Injectable } from '@nestjs/common';
import { RedisService } from '@/common/redis/redis.service';
import Redis from 'ioredis';

@Injectable()
export class RedisTokenService {
    private readonly redis: Redis;
    private readonly tokenRevocationPrefix = 'blacklist:token';
    private readonly sessionRevocationPrefix = 'revoked:session';

    constructor(private readonly redisService: RedisService) {
        this.redis = redisService.getClient();
    }

    async set(key: string, value: string, ttlSeconds: number): Promise<void> {
        try {
            await this.redis.set(key, value, 'EX', ttlSeconds);
            this.redisService.recordSuccess();
        } catch (error) {
            this.redisService.recordFailure();
            throw error;
        }
    }

    async get(key: string): Promise<string | null> {
        try {
            const val = await this.redis.get(key);
            this.redisService.recordSuccess();
            return val;
        } catch (error) {
            this.redisService.recordFailure();
            throw error;
        }
    }

    async del(key: string): Promise<void> {
        try {
            await this.redis.del(key);
            this.redisService.recordSuccess();
        } catch (error) {
            this.redisService.recordFailure();
            throw error;
        }
    }

    async revokeToken(jti: string, remainingTtlSeconds: number): Promise<void> {
        if (remainingTtlSeconds <= 0) return;
        await this.set(this.getTokenRevocationKey(jti), 'true', remainingTtlSeconds);
    }

    async isTokenRevoked(jti: string): Promise<boolean> {
        const value = await this.get(this.getTokenRevocationKey(jti));
        return value === 'true';
    }

    async revokeSession(sessionId: string, ttlSeconds: number): Promise<void> {
        if (ttlSeconds <= 0) return;
        await this.set(this.getSessionRevocationKey(sessionId), 'true', ttlSeconds);
    }

    async isSessionRevoked(sessionId: string): Promise<boolean> {
        try {
            const exists = await this.redis.exists(this.getSessionRevocationKey(sessionId));
            this.redisService.recordSuccess();
            return exists === 1;
        } catch (error) {
            this.redisService.recordFailure();
            throw error;
        }
    }

    private getTokenRevocationKey(jti: string): string {
        return `${this.tokenRevocationPrefix}:${jti}`;
    }

    private getSessionRevocationKey(sessionId: string): string {
        return `${this.sessionRevocationPrefix}:${sessionId}`;
    }
}
