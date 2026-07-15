import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { LoginChallengeSelect } from '@/module-auth-token/schemas/login-challenges.schema';
import { createHash } from 'crypto';

describe('LoginChallengeService', () => {
    let service: LoginChallengeService;
    let mockRepository: jest.Mocked<LoginChallengeRepository>;
    const now = new Date('2026-06-23T12:00:00Z');

    const buildMockChallenge = (
        overrides: Partial<LoginChallengeSelect> = {},
    ): LoginChallengeSelect => ({
        id: overrides.id ?? 'challenge-uuid',
        userId: overrides.userId ?? 'user-uuid',
        realm: overrides.realm ?? 'customer',
        knownDeviceId: overrides.knownDeviceId ?? 'known-device-uuid',
        deviceId: overrides.deviceId ?? 'device-uuid',
        challengeType: overrides.challengeType ?? 'email_code',
        checkpointTokenHash: overrides.checkpointTokenHash ?? 'token-hash',
        codeHash: overrides.codeHash ?? 'code-hash',
        attemptCount: overrides.attemptCount ?? 0,
        maxAttempts: overrides.maxAttempts ?? 5,
        expiresAt: overrides.expiresAt ?? new Date(now.getTime() + 300000),
        approvedAt: overrides.approvedAt ?? null,
        consumedAt: overrides.consumedAt ?? null,
        failedAt: overrides.failedAt ?? null,
        expiredAt: overrides.expiredAt ?? null,
        createdAt: overrides.createdAt ?? now,
        ipAddress: overrides.ipAddress ?? null,
        country: overrides.country ?? null,
        region: overrides.region ?? null,
        city: overrides.city ?? null,
        userAgent: overrides.userAgent ?? null,
        riskScore: overrides.riskScore ?? 0,
        riskReason: overrides.riskReason ?? null,
    });

    beforeEach(() => {
        mockRepository = {
            createSafe: jest.fn(),
            findById: jest.fn(),
            findByTokenHash: jest.fn(),
            findLatestByUserRealmDevice: jest.fn(),
            consume: jest.fn(),
            approveAndConsume: jest.fn(),
            fail: jest.fn(),
            incrementAttempts: jest.fn(),
            expire: jest.fn(),
            approveAndConsumeAtomically: jest.fn(),
        } as unknown as jest.Mocked<LoginChallengeRepository>;

        service = new LoginChallengeService(mockRepository);
    });

    describe('createLoginChallenge', () => {
        it('generates plaintext code and token, hashes them, and persists safely', async () => {
            const expectedRow = buildMockChallenge();
            mockRepository.createSafe.mockResolvedValue(expectedRow);

            const result = await service.createLoginChallenge(
                'user-uuid',
                'customer',
                'known-device-uuid',
                'device-uuid',
                {
                    ipAddress: '127.0.0.1',
                    userAgent: 'Mozilla/5.0',
                    country: 'UA',
                    region: 'Kyiv',
                    city: 'Kyiv',
                },
                {
                    riskScore: 20,
                    riskReason: 'New Device',
                },
                now,
            );

            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.createSafe).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: 'user-uuid',
                    realm: 'customer',
                    knownDeviceId: 'known-device-uuid',
                    deviceId: 'device-uuid',
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
                    checkpointTokenHash: expect.any(String),
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
                    codeHash: expect.any(String),
                    expiresAt: new Date(now.getTime() + 5 * 60 * 1000),
                    ipAddress: '127.0.0.1',
                    userAgent: 'Mozilla/5.0',
                    country: 'UA',
                    region: 'Kyiv',
                    city: 'Kyiv',
                    riskScore: 20,
                    riskReason: 'New Device',
                    createdAt: now,
                }),
                now,
                undefined,
            );

            // Plaintext values must be returned
            expect(result.challenge).toEqual(expectedRow);
            expect(result.checkpointToken).toBeDefined();
            expect(result.code).toHaveLength(6);

            // Plaintext values must match hashes sent to database
            const expectedTokenHash = createHash('sha256')
                .update(result.checkpointToken)
                .digest('hex');
            const expectedCodeHash = createHash('sha256').update(result.code).digest('hex');

            const createSafeArgs = mockRepository.createSafe.mock.calls[0][0];
            expect(createSafeArgs.checkpointTokenHash).toBe(expectedTokenHash);
            expect(createSafeArgs.codeHash).toBe(expectedCodeHash);
        });
    });

    describe('verifyLoginChallenge', () => {
        it('returns true on valid active challenge and matching code', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const challenge = buildMockChallenge({ codeHash });

            mockRepository.findById.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.findById).toHaveBeenCalledWith(challenge.id);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.incrementAttempts).not.toHaveBeenCalled();
        });

        it('returns false and increments attempts on invalid code', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update('wrong-code').digest('hex');
            const challenge = buildMockChallenge({ id: 'challenge-id', codeHash });

            mockRepository.findById.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(false);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.incrementAttempts).toHaveBeenCalledWith('challenge-id', now);
        });

        it('returns false and transitions state to expired in DB if challenge is expired', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const challenge = buildMockChallenge({
                codeHash,
                expiresAt: new Date(now.getTime() - 1000), // expired 1s ago
            });

            mockRepository.findById.mockResolvedValue(challenge);
            mockRepository.expire.mockResolvedValue(true);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(false);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.expire).toHaveBeenCalledWith(challenge.id, now);
        });

        it('returns false if challenge is already consumed', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const challenge = buildMockChallenge({
                codeHash,
                consumedAt: now,
            });

            mockRepository.findById.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(false);
        });

        it('returns false if challenge is already failed', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const challenge = buildMockChallenge({
                codeHash,
                failedAt: now,
            });

            mockRepository.findById.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(false);
        });

        it('returns false if max attempts reached', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const challenge = buildMockChallenge({
                codeHash,
                attemptCount: 5,
                maxAttempts: 5,
            });

            mockRepository.findById.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(challenge.id, code, now);

            expect(result).toBe(false);
        });

        it('falls back to lookup by token hash if search by UUID fails', async () => {
            const code = '123456';
            const codeHash = createHash('sha256').update(code).digest('hex');
            const token = 'checkpoint-token-string';
            const tokenHash = createHash('sha256').update(token).digest('hex');
            const challenge = buildMockChallenge({ codeHash, checkpointTokenHash: tokenHash });

            mockRepository.findById.mockResolvedValue(null);
            mockRepository.findByTokenHash.mockResolvedValue(challenge);

            const result = await service.verifyLoginChallenge(token, code, now);

            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.findById).toHaveBeenCalledWith(token);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.findByTokenHash).toHaveBeenCalledWith(tokenHash);
        });
    });

    describe('getLatestChallengeForDevice', () => {
        it('proxies latest challenge lookup to repository', async () => {
            const challenge = buildMockChallenge();
            mockRepository.findLatestByUserRealmDevice.mockResolvedValue(challenge);

            const result = await service.getLatestChallengeForDevice(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(result).toEqual(challenge);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.findLatestByUserRealmDevice).toHaveBeenCalledWith(
                'user-uuid',
                'customer',
                'device-uuid',
                undefined,
            );
        });
    });

    describe('getResendEligibility', () => {
        it('returns eligible active state for an active latest challenge inside resend window', () => {
            const challenge = buildMockChallenge({
                createdAt: new Date(now.getTime() - 60_000),
                expiresAt: new Date(now.getTime() + 60_000),
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: true,
                state: 'active',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns eligible expired state for an expired latest challenge inside resend window', () => {
            const challenge = buildMockChallenge({
                createdAt: new Date(now.getTime() - 6 * 60 * 1000),
                expiresAt: new Date(now.getTime() - 60_000),
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: true,
                state: 'expired',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns consumed for consumed challenge', () => {
            const challenge = buildMockChallenge({
                consumedAt: new Date(now.getTime() - 1000),
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: false,
                reason: 'consumed',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns failed for failed challenge', () => {
            const challenge = buildMockChallenge({
                failedAt: new Date(now.getTime() - 1000),
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: false,
                reason: 'failed',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns failed for challenge that reached max attempts', () => {
            const challenge = buildMockChallenge({
                attemptCount: 5,
                maxAttempts: 5,
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: false,
                reason: 'failed',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns superseded when the challenge is not the latest for its device', () => {
            const challenge = buildMockChallenge({ id: 'old-challenge' });
            const latestChallenge = buildMockChallenge({ id: 'new-challenge' });

            const result = service.getResendEligibility(challenge, now, latestChallenge);

            expect(result).toEqual({
                eligible: false,
                reason: 'superseded',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });

        it('returns outside_resend_window for challenge created outside the resend window', () => {
            const challenge = buildMockChallenge({
                createdAt: new Date(now.getTime() - 16 * 60 * 1000),
                expiresAt: new Date(now.getTime() - 11 * 60 * 1000),
            });

            const result = service.getResendEligibility(challenge, now, challenge);

            expect(result).toEqual({
                eligible: false,
                reason: 'outside_resend_window',
                resendWindowExpiresAt: new Date(challenge.createdAt.getTime() + 15 * 60 * 1000),
            });
        });
    });

    describe('state mutators', () => {
        it('consumeLoginChallenge proxies to repository', async () => {
            mockRepository.consume.mockResolvedValue(true);
            const result = await service.consumeLoginChallenge('id', now);
            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.consume).toHaveBeenCalledWith('id', now);
        });

        it('failLoginChallenge proxies to repository', async () => {
            mockRepository.fail.mockResolvedValue(true);
            const result = await service.failLoginChallenge('id', now);
            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.fail).toHaveBeenCalledWith('id', now);
        });

        it('expireLoginChallenge proxies to repository', async () => {
            mockRepository.expire.mockResolvedValue(true);
            const result = await service.expireLoginChallenge('id', now);
            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.expire).toHaveBeenCalledWith('id', now);
        });

        it('approveAndConsumeLoginChallengeAtomically proxies to repository', async () => {
            const issueAuthState = jest.fn().mockResolvedValue({ token: 'test' });
            mockRepository.approveAndConsumeAtomically.mockResolvedValue({ token: 'test' });

            const result = await service.approveAndConsumeLoginChallengeAtomically(
                'id',
                issueAuthState,
                now,
            );

            expect(result).toEqual({ token: 'test' });
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockRepository.approveAndConsumeAtomically).toHaveBeenCalledWith(
                'id',
                issueAuthState,
                now,
            );
        });
    });
});
