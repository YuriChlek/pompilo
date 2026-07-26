import { Injectable, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis, { RedisOptions } from 'ioredis';

@Injectable()
export class RedisService implements OnModuleDestroy {
    private readonly client: Redis;
    private consecutiveFailures = 0;
    private readonly maxConsecutiveFailures: number;

    constructor(configService: ConfigService) {
        const password = configService.get<string>('REDIS_PASSWORD');
        const db = Number(configService.get<string>('REDIS_DB') ?? '0');
        const options: RedisOptions = {
            host: configService.get<string>('REDIS_HOST') ?? 'localhost',
            port: Number(configService.get<string>('REDIS_PORT') ?? '6379'),
            db,
        };

        if (password) {
            options.password = password;
        }

        this.maxConsecutiveFailures = Number(
            configService.get<string>('REDIS_MAX_CONSECUTIVE_FAILURES') ?? '5',
        );

        this.client = new Redis(options);
    }

    getClient(): Redis {
        return this.client;
    }

    recordSuccess(): void {
        this.consecutiveFailures = 0;
    }

    recordFailure(): void {
        this.consecutiveFailures += 1;
    }

    isHealthy(): boolean {
        return this.consecutiveFailures < this.maxConsecutiveFailures;
    }

    getConsecutiveFailures(): number {
        return this.consecutiveFailures;
    }

    async onModuleDestroy(): Promise<void> {
        try {
            await this.client.quit();
        } catch {
            this.client.disconnect();
        }
    }
}
