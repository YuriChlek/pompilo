import { Test, TestingModule } from '@nestjs/testing';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { RedisService } from '@/common/redis/redis.service';

describe('MailRedisService', () => {
    let service: MailRedisService;
    let redisClient: {
        get: jest.Mock;
        set: jest.Mock;
        del: jest.Mock;
        incr: jest.Mock;
        expire: jest.Mock;
        lpush: jest.Mock;
        ltrim: jest.Mock;
        lrange: jest.Mock;
    };

    beforeEach(async () => {
        redisClient = {
            get: jest.fn(),
            set: jest.fn(),
            del: jest.fn(),
            incr: jest.fn(),
            expire: jest.fn(),
            lpush: jest.fn(),
            ltrim: jest.fn(),
            lrange: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailRedisService,
                {
                    provide: RedisService,
                    useValue: { getClient: () => redisClient },
                },
            ],
        }).compile();

        service = module.get<MailRedisService>(MailRedisService);
    });

    it('should set readiness status with TTL', async () => {
        await service.setReadinessStatus('unhealthy', 'error message', 100);
        expect(redisClient.set).toHaveBeenCalledWith(
            'mail:readiness_status',
            expect.stringContaining('"status":"unhealthy","error":"error message"'),
            'EX',
            100,
        );
    });

    it('should get readiness status', async () => {
        const data = JSON.stringify({
            status: 'unhealthy',
            error: 'some error',
            timestamp: new Date().toISOString(),
        });
        redisClient.get.mockResolvedValue(data);

        const result = await service.getReadinessStatus();
        expect(result?.status).toBe('unhealthy');
        expect(result?.error).toBe('some error');
    });

    it('should return null if readiness status is not in Redis', async () => {
        redisClient.get.mockResolvedValue(null);
        const result = await service.getReadinessStatus();
        expect(result).toBeNull();
    });

    it('should increment failure count and set TTL', async () => {
        redisClient.incr.mockResolvedValue(1);
        const result = await service.incrementFailureCount();
        expect(result).toBe(1);
        expect(redisClient.incr).toHaveBeenCalledWith('mail:failure_count');
        expect(redisClient.expire).toHaveBeenCalledWith('mail:failure_count', 1800);
    });

    it('should log successful delivery and reset everything', async () => {
        await service.logSuccessfulDelivery();
        expect(redisClient.set).toHaveBeenCalledWith(
            'mail:last_successful_send',
            expect.any(String),
        );
        expect(redisClient.del).toHaveBeenCalledWith('mail:failure_count');
        expect(redisClient.del).toHaveBeenCalledWith('mail:delivery_errors');
    });
});
