import {
    loginChallenges,
    isLoginChallengeActive,
    isLoginChallengeFailed,
    isLoginChallengeExpired,
    isLoginChallengeConsumed,
    isLoginChallengeApproved,
    LoginChallengeSelect,
} from '@/module-auth-token/schemas/login-challenges.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

const buildMockChallenge = (
    overrides: Partial<LoginChallengeSelect> = {},
): LoginChallengeSelect => ({
    id: overrides.id ?? 'challenge-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    knownDeviceId: overrides.knownDeviceId ?? 'device-uuid',
    deviceId: overrides.deviceId ?? 'device-uuid-123',
    challengeType: overrides.challengeType ?? 'email_code',
    checkpointTokenHash: overrides.checkpointTokenHash ?? 'token-hash',
    codeHash: overrides.codeHash ?? 'code-hash',
    attemptCount: overrides.attemptCount ?? 0,
    maxAttempts: overrides.maxAttempts ?? 5,
    expiresAt: overrides.expiresAt ?? new Date(Date.now() + 600000), // 10 mins in future
    approvedAt: overrides.approvedAt ?? null,
    consumedAt: overrides.consumedAt ?? null,
    failedAt: overrides.failedAt ?? null,
    expiredAt: overrides.expiredAt ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    ipAddress: overrides.ipAddress ?? null,
    country: overrides.country ?? null,
    region: overrides.region ?? null,
    city: overrides.city ?? null,
    userAgent: overrides.userAgent ?? null,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
});

describe('LoginChallengesSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(loginChallenges);
        expect(config.name).toBe('login_challenges');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(loginChallenges);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('login_challenge_id');
        expect(columnNames).toContain('user_id');
        expect(columnNames).toContain('realm');
        expect(columnNames).toContain('known_device_id');
        expect(columnNames).toContain('device_id');
        expect(columnNames).toContain('challenge_type');
        expect(columnNames).toContain('checkpoint_token_hash');
        expect(columnNames).toContain('code_hash');
        expect(columnNames).toContain('attempt_count');
        expect(columnNames).toContain('max_attempts');
        expect(columnNames).toContain('expires_at');
        expect(columnNames).toContain('approved_at');
        expect(columnNames).toContain('consumed_at');
        expect(columnNames).toContain('failed_at');
        expect(columnNames).toContain('expired_at');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('ip_address');
        expect(columnNames).toContain('country');
        expect(columnNames).toContain('region');
        expect(columnNames).toContain('city');
        expect(columnNames).toContain('user_agent');
        expect(columnNames).toContain('risk_score');
        expect(columnNames).toContain('risk_reason');
    });

    it('should have correct foreign keys referencing users and known_devices with cascade deletion', () => {
        const config = getTableConfig(loginChallenges);
        expect(config.foreignKeys.length).toBeGreaterThanOrEqual(2);

        const userFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'user_id'),
        );
        expect(userFk).toBeDefined();
        expect(userFk?.onDelete).toBe('cascade');

        const deviceFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'known_device_id'),
        );
        expect(deviceFk).toBeDefined();
        expect(deviceFk?.onDelete).toBe('cascade');
    });

    it('should have correct indexes including partial unique index', () => {
        const config = getTableConfig(loginChallenges);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('login_challenges_active_unique');
        expect(indexNames).toContain('login_challenges_expiry_idx');

        const activeUniqueIdx = indexes.find(
            idx => idx.config.name === 'login_challenges_active_unique',
        );
        expect(activeUniqueIdx).toBeDefined();
        expect(activeUniqueIdx?.config.unique).toBe(true);

        const expiryIdx = indexes.find(idx => idx.config.name === 'login_challenges_expiry_idx');
        expect(expiryIdx).toBeDefined();
        expect(expiryIdx?.config.unique).toBeFalsy();
    });

    it('should enforce realm and attempt-count invariants', () => {
        const config = getTableConfig(loginChallenges);
        const checkNames = config.checks.map(check => check.name);

        expect(checkNames).toContain('login_challenges_realm_check');
        expect(checkNames).toContain('login_challenges_attempts_check');
    });

    describe('State Predicates', () => {
        it('should correctly identify active challenge', () => {
            const challenge = buildMockChallenge();
            expect(isLoginChallengeActive(challenge)).toBe(true);
            expect(isLoginChallengeFailed(challenge)).toBe(false);
            expect(isLoginChallengeExpired(challenge)).toBe(false);
            expect(isLoginChallengeConsumed(challenge)).toBe(false);
            expect(isLoginChallengeApproved(challenge)).toBe(false);
        });

        it('should identify as failed when failedAt is set or attempts exceeded', () => {
            const challengeWithFailedAt = buildMockChallenge({ failedAt: new Date() });
            expect(isLoginChallengeFailed(challengeWithFailedAt)).toBe(true);
            expect(isLoginChallengeActive(challengeWithFailedAt)).toBe(false);

            const challengeWithExceededAttempts = buildMockChallenge({ attemptCount: 5 });
            expect(isLoginChallengeFailed(challengeWithExceededAttempts)).toBe(true);
            expect(isLoginChallengeActive(challengeWithExceededAttempts)).toBe(false);
        });

        it('should identify as expired when expiredAt is set or current time is past expiresAt', () => {
            const challengeWithExpiredAt = buildMockChallenge({ expiredAt: new Date() });
            expect(isLoginChallengeExpired(challengeWithExpiredAt)).toBe(true);
            expect(isLoginChallengeActive(challengeWithExpiredAt)).toBe(false);

            const pastDate = new Date(Date.now() - 1000);
            const challengeExpiredByTime = buildMockChallenge({ expiresAt: pastDate });
            expect(isLoginChallengeExpired(challengeExpiredByTime, new Date())).toBe(true);
            expect(isLoginChallengeActive(challengeExpiredByTime, new Date())).toBe(false);
        });

        it('should identify as consumed when consumedAt is set', () => {
            const challenge = buildMockChallenge({ consumedAt: new Date() });
            expect(isLoginChallengeConsumed(challenge)).toBe(true);
            expect(isLoginChallengeActive(challenge)).toBe(false);
        });

        it('should identify as approved when approvedAt is set', () => {
            const challenge = buildMockChallenge({ approvedAt: new Date() });
            expect(isLoginChallengeApproved(challenge)).toBe(true);
        });
    });
});
