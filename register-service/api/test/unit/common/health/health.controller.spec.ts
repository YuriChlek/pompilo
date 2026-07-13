import { Test, TestingModule } from '@nestjs/testing';
import { HealthController } from '@/common/health/health.controller';
import { RedisService } from '@/common/redis/redis.service';
import { ServiceUnavailableException } from '@nestjs/common';

describe('HealthController', () => {
    let controller: HealthController;
    let redisService: {
        isHealthy: jest.Mock;
        getClient: jest.Mock;
        recordSuccess: jest.Mock;
        recordFailure: jest.Mock;
    };

    beforeEach(async () => {
        redisService = {
            isHealthy: jest.fn().mockReturnValue(true),
            getClient: jest.fn(),
            recordSuccess: jest.fn(),
            recordFailure: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [HealthController],
            providers: [
                {
                    provide: RedisService,
                    useValue: redisService,
                },
            ],
        }).compile();

        controller = module.get<HealthController>(HealthController);
    });

    it('should return healthy if Redis is healthy and ping succeeds', async () => {
        const pingMock = jest.fn().mockResolvedValue('PONG');
        redisService.getClient.mockReturnValue({
            ping: pingMock,
        });

        const result = await controller.checkReadiness();

        expect(result).toEqual({ status: 'healthy' });
        expect(redisService.isHealthy).toHaveBeenCalled();
        expect(redisService.recordSuccess).toHaveBeenCalled();
    });

    it('should throw ServiceUnavailableException if Redis consecutive failures exceed limit', async () => {
        redisService.isHealthy.mockReturnValue(false);

        await expect(controller.checkReadiness()).rejects.toThrow(ServiceUnavailableException);

        expect(redisService.isHealthy).toHaveBeenCalled();
        expect(redisService.getClient).not.toHaveBeenCalled();
    });

    it('should throw ServiceUnavailableException and record failure if ping fails', async () => {
        const pingMock = jest.fn().mockRejectedValue(new Error('Connection lost'));
        redisService.getClient.mockReturnValue({
            ping: pingMock,
        });

        await expect(controller.checkReadiness()).rejects.toThrow(ServiceUnavailableException);

        expect(redisService.isHealthy).toHaveBeenCalled();
        expect(redisService.recordFailure).toHaveBeenCalled();
    });
});
