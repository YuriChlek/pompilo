import { LoginChallengeResendPolicyService } from '@/module-auth-token/services/login-challenge-resend-policy.service';
import { RedisService } from '@/common/redis/redis.service';
import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { LoginChallengeSelect } from '@/module-auth-token/schemas/login-challenges.schema';

describe('LoginChallengeResendPolicyService', () => {
    const input = {
        userId: 'user-uuid',
        realm: 'customer',
        deviceId: 'device-uuid',
        ipAddress: '203.0.113.10',
    };

    let redis: {
        eval: jest.Mock;
        pttl: jest.Mock;
        set: jest.Mock;
    };
    let redisService: {
        getClient: jest.Mock;
        recordSuccess: jest.Mock;
        recordFailure: jest.Mock;
    };
    let service: LoginChallengeResendPolicyService;

    beforeEach(() => {
        redis = {
            eval: jest.fn().mockResolvedValue(1),
            pttl: jest.fn().mockResolvedValue(-2),
            set: jest.fn().mockResolvedValue('OK'),
        };
        redisService = {
            getClient: jest.fn().mockReturnValue(redis),
            recordSuccess: jest.fn(),
            recordFailure: jest.fn(),
        };

        service = new LoginChallengeResendPolicyService(redisService as unknown as RedisService);
    });

    it('reserves allowed resend attempt and sets cooldown', async () => {
        const result = await service.reserveResendAttempt(input);

        expect(result).toEqual({ allowed: true, retryAfterSeconds: 0 });
        expect(redis.eval).toHaveBeenCalledTimes(2);
        expect(redis.set).toHaveBeenCalledTimes(1);
        expect(redis.set).toHaveBeenCalledWith(
            expect.stringContaining('login-challenge:resend:cooldown:'),
            '1',
            'PX',
            60_000,
        );
        expect(redisService.recordSuccess).toHaveBeenCalledTimes(1);
    });

    it('blocks resend during cooldown without incrementing rate buckets', async () => {
        redis.pttl.mockResolvedValueOnce(42_000);

        const result = await service.reserveResendAttempt(input);

        expect(result).toEqual({
            allowed: false,
            reason: 'cooldown',
            retryAfterSeconds: 42,
        });
        expect(redis.eval).not.toHaveBeenCalled();
        expect(redis.set).not.toHaveBeenCalled();
    });

    it('blocks resend when user device window exceeds max attempts', async () => {
        redis.eval.mockResolvedValueOnce(6);
        redis.pttl.mockResolvedValueOnce(-2).mockResolvedValueOnce(121_000);

        const result = await service.reserveResendAttempt(input);

        expect(result).toEqual({
            allowed: false,
            reason: 'user_device_rate_limited',
            retryAfterSeconds: 121,
        });
        expect(redis.eval).toHaveBeenCalledTimes(1);
        expect(redis.set).not.toHaveBeenCalled();
    });

    it('blocks resend when IP window exceeds max attempts', async () => {
        redis.eval.mockResolvedValueOnce(1).mockResolvedValueOnce(21);
        redis.pttl.mockResolvedValueOnce(-2).mockResolvedValueOnce(75_000);

        const result = await service.reserveResendAttempt(input);

        expect(result).toEqual({
            allowed: false,
            reason: 'ip_rate_limited',
            retryAfterSeconds: 75,
        });
        expect(redis.eval).toHaveBeenCalledTimes(2);
        expect(redis.set).not.toHaveBeenCalled();
    });

    it('uses hashed Redis keys without plaintext identifiers', async () => {
        await service.reserveResendAttempt(input);

        const evalCalls = redis.eval.mock.calls as unknown[][];
        const setCalls = redis.set.mock.calls as unknown[][];
        const keys = [evalCalls[0][2], evalCalls[1][2], setCalls[0][0]] as string[];

        for (const key of keys) {
            expect(key).not.toContain(input.userId);
            expect(key).not.toContain(input.realm);
            expect(key).not.toContain(input.deviceId);
            expect(key).not.toContain(input.ipAddress);
        }
    });

    it('records Redis failure and rethrows', async () => {
        redis.pttl.mockRejectedValueOnce(new Error('Redis unavailable'));

        await expect(service.reserveResendAttempt(input)).rejects.toThrow('Redis unavailable');
        expect(redisService.recordFailure).toHaveBeenCalledTimes(1);
    });
});

describe('LoginChallengeService resend policy isolation', () => {
    it('does not touch resend policy counters during initial challenge creation', async () => {
        const challenge: LoginChallengeSelect = {
            id: 'challenge-uuid',
            userId: 'user-uuid',
            realm: 'customer',
            knownDeviceId: 'known-device-uuid',
            deviceId: 'device-uuid',
            challengeType: 'email_code',
            checkpointTokenHash: 'token-hash',
            codeHash: 'code-hash',
            attemptCount: 0,
            maxAttempts: 5,
            expiresAt: new Date('2026-06-23T12:05:00Z'),
            approvedAt: null,
            consumedAt: null,
            failedAt: null,
            expiredAt: null,
            createdAt: new Date('2026-06-23T12:00:00Z'),
            ipAddress: null,
            country: null,
            region: null,
            city: null,
            userAgent: null,
            riskScore: 0,
            riskReason: null,
        };
        const repository = {
            createSafe: jest.fn().mockResolvedValue(challenge),
        } as unknown as jest.Mocked<LoginChallengeRepository>;
        const service = new LoginChallengeService(repository);

        await service.createLoginChallenge(
            'user-uuid',
            'customer',
            'known-device-uuid',
            'device-uuid',
            {},
            { riskScore: 0 },
            new Date('2026-06-23T12:00:00Z'),
        );

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(repository.createSafe).toHaveBeenCalledTimes(1);
        expect(Object.keys(repository)).toEqual(['createSafe']);
    });
});
