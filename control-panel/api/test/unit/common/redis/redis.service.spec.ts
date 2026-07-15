import { Test, TestingModule } from '@nestjs/testing';
import { RedisService } from '@/common/redis/redis.service';
import { ConfigService } from '@nestjs/config';

// Mock ioredis to avoid actual Redis connection during tests
jest.mock('ioredis', () => {
    return jest.fn().mockImplementation(() => {
        return {
            quit: jest.fn().mockResolvedValue('OK'),
            disconnect: jest.fn(),
        };
    });
});

describe('RedisService', () => {
    let service: RedisService;
    let configService: {
        get: jest.Mock;
    };

    beforeEach(async () => {
        configService = {
            get: jest.fn().mockImplementation((key: string) => {
                if (key === 'REDIS_MAX_CONSECUTIVE_FAILURES') return '3';
                return undefined;
            }),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                RedisService,
                {
                    provide: ConfigService,
                    useValue: configService,
                },
            ],
        }).compile();

        service = module.get<RedisService>(RedisService);
    });

    it('should be defined and healthy initially', () => {
        expect(service).toBeDefined();
        expect(service.isHealthy()).toBe(true);
        expect(service.getConsecutiveFailures()).toBe(0);
    });

    it('should track consecutive failures and become unhealthy', () => {
        service.recordFailure();
        expect(service.getConsecutiveFailures()).toBe(1);
        expect(service.isHealthy()).toBe(true);

        service.recordFailure();
        expect(service.getConsecutiveFailures()).toBe(2);
        expect(service.isHealthy()).toBe(true);

        service.recordFailure();
        expect(service.getConsecutiveFailures()).toBe(3);
        expect(service.isHealthy()).toBe(false); // unhealthy since limit is 3
    });

    it('should reset consecutive failures on success', () => {
        service.recordFailure();
        service.recordFailure();
        expect(service.getConsecutiveFailures()).toBe(2);

        service.recordSuccess();
        expect(service.getConsecutiveFailures()).toBe(0);
        expect(service.isHealthy()).toBe(true);
    });
});
