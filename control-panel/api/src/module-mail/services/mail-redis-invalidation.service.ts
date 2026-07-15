import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';

@Injectable()
export class MailRedisInvalidationService implements OnModuleInit, OnModuleDestroy {
    private readonly logger = new Logger(MailRedisInvalidationService.name);
    private readonly CHANNEL = 'mail:config_invalidation';
    private subClient: Redis | null = null;
    private pubClient: Redis | null = null;
    private callback: (() => void) | null = null;

    constructor(private readonly configService: ConfigService) {}

    async onModuleInit() {
        const host = this.configService.get<string>('REDIS_HOST') ?? 'localhost';
        const port = Number(this.configService.get<string>('REDIS_PORT') ?? '6379');
        const password = this.configService.get<string>('REDIS_PASSWORD');

        const redisOptions = { host, port, password };

        this.subClient = new Redis(redisOptions);
        this.pubClient = new Redis(redisOptions);

        await this.subClient.subscribe(this.CHANNEL);

        this.subClient.on('message', (channel, message) => {
            if (channel === this.CHANNEL) {
                this.logger.log(`Received invalidation signal: ${message}`);
                if (this.callback) {
                    this.callback();
                }
            }
        });

        this.logger.log(`Subscribed to ${this.CHANNEL}`);
    }

    async onModuleDestroy() {
        await this.subClient?.quit();
        await this.pubClient?.quit();
    }

    onInvalidate(callback: () => void) {
        this.callback = callback;
    }

    async invalidate() {
        if (this.pubClient) {
            await this.pubClient.publish(this.CHANNEL, `invalidate:${Date.now()}`);
        }
    }
}
