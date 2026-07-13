import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { MailRedisInvalidationService } from '@/module-mail/services/mail-redis-invalidation.service';
import Redis from 'ioredis';

jest.mock('ioredis');

describe('MailRedisInvalidationService', () => {
    let service: MailRedisInvalidationService;
    let configService: {
        get: jest.Mock;
    };
    let mockRedis: {
        subscribe: jest.Mock;
        on: jest.Mock;
        publish: jest.Mock;
        quit: jest.Mock;
    };

    beforeEach(async () => {
        configService = {
            get: jest.fn(),
        };

        mockRedis = {
            subscribe: jest.fn().mockResolvedValue(undefined),
            on: jest.fn(),
            publish: jest.fn().mockResolvedValue(1),
            quit: jest.fn().mockResolvedValue('OK'),
        };

        (Redis as unknown as jest.Mock).mockReturnValue(mockRedis);

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailRedisInvalidationService,
                { provide: ConfigService, useValue: configService },
            ],
        }).compile();

        service = module.get<MailRedisInvalidationService>(MailRedisInvalidationService);
    });

    it('should subscribe to channel on init', async () => {
        await service.onModuleInit();
        expect(mockRedis.subscribe).toHaveBeenCalledWith('mail:config_invalidation');
    });

    it('should call callback when message is received', async () => {
        const callback = jest.fn();
        service.onInvalidate(callback);

        await service.onModuleInit();

        // Trigger message event
        const onCalls = mockRedis.on.mock.calls as unknown[][];
        const onMessageCall = onCalls.find(c => c[0] === 'message');
        const onMessage = onMessageCall
            ? (onMessageCall[1] as (channel: string, message: string) => void)
            : null;
        if (onMessage) {
            onMessage('mail:config_invalidation', 'invalidate:123');
        }

        expect(callback).toHaveBeenCalled();
    });

    it('should publish message on invalidate', async () => {
        await service.onModuleInit();
        await service.invalidate();
        expect(mockRedis.publish).toHaveBeenCalledWith(
            'mail:config_invalidation',
            expect.stringContaining('invalidate:'),
        );
    });
});
