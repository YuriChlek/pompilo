import { EmailFlowRateLimitService } from '@/common/rate-limiting/services/email-flow-rate-limit.service';
import { RedisService } from '@/common/redis/redis.service';

describe('EmailFlowRateLimitService', () => {
    let redis: {
        eval: jest.Mock;
        pttl: jest.Mock;
    };
    let service: EmailFlowRateLimitService;

    beforeEach(() => {
        redis = {
            eval: jest.fn().mockResolvedValue(1),
            pttl: jest.fn().mockResolvedValue(120000),
        };

        service = new EmailFlowRateLimitService({
            getClient: () => redis,
        } as unknown as RedisService);
    });

    it('uses Redis-backed IP and recipient buckets without plaintext identifiers in keys', async () => {
        await service.check({
            flow: 'registration',
            ipAddress: '203.0.113.10',
            recipientEmail: 'Victim@Example.com ',
        });

        expect(redis.eval).toHaveBeenCalledTimes(2);

        const calls = redis.eval.mock.calls as unknown[][];
        const ipKey = calls[0][2] as string;
        const recipientKey = calls[1][2] as string;

        expect(ipKey).toContain('rate-limit:email-flow:registration:ip:');
        expect(recipientKey).toContain('rate-limit:email-flow:registration:recipient:');
        expect(ipKey).not.toContain('203.0.113.10');
        expect(recipientKey).not.toContain('Victim');
        expect(recipientKey).not.toContain('example.com');
    });

    it('returns limited with retry-after when any bucket exceeds its configured limit', async () => {
        redis.eval.mockResolvedValueOnce(1).mockResolvedValueOnce(4);
        redis.pttl.mockResolvedValue(61000);

        const result = await service.check({
            flow: 'password_reset',
            ipAddress: '203.0.113.10',
            recipientEmail: 'victim@example.com',
        });

        expect(result).toEqual({ limited: true, retryAfterSeconds: 61 });
        expect(redis.pttl).toHaveBeenCalledTimes(1);
    });

    it('checks only the IP bucket when request has no recipient email yet', async () => {
        const result = await service.check({
            flow: 'email_change',
            ipAddress: '203.0.113.10',
        });

        expect(result).toEqual({ limited: false, retryAfterSeconds: 0 });
        expect(redis.eval).toHaveBeenCalledTimes(1);
    });
});
