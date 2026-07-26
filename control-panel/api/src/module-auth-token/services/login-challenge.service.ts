import { Injectable } from '@nestjs/common';
import { createHash, randomInt, randomUUID } from 'crypto';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { LoginChallengeSelect } from '@/module-auth-token/schemas/login-challenges.schema';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import {
    LOGIN_CHALLENGE_RESEND_WINDOW_MS,
    LOGIN_CHALLENGE_TTL_MS,
} from '@/module-auth-token/constants/login-challenge.constants';
import type { LoginChallengeResendEligibility } from '@/module-auth-token/types/login-challenge.types';

@Injectable()
export class LoginChallengeService {
    constructor(private readonly loginChallengeRepository: LoginChallengeRepository) {}

    private hash(value: string): string {
        return createHash('sha256').update(value).digest('hex');
    }

    async createLoginChallenge(
        userId: string,
        realm: AuthRealm,
        knownDeviceId: string | null,
        deviceId: string,
        metadata: {
            ipAddress?: string;
            userAgent?: string;
            country?: string;
            region?: string;
            city?: string;
        },
        risk: {
            riskScore: number;
            riskReason?: string;
        },
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<{ challenge: LoginChallengeSelect; checkpointToken: string; code: string }> {
        const checkpointToken = randomUUID();
        const code = randomInt(100000, 1000000).toString();

        const checkpointTokenHash = this.hash(checkpointToken);
        const codeHash = this.hash(code);

        const expiresAt = new Date(now.getTime() + LOGIN_CHALLENGE_TTL_MS);

        const challenge = await this.loginChallengeRepository.createSafe(
            {
                userId,
                realm,
                knownDeviceId,
                deviceId,
                checkpointTokenHash,
                codeHash,
                expiresAt,
                ipAddress: metadata.ipAddress ?? null,
                userAgent: metadata.userAgent ?? null,
                country: metadata.country ?? null,
                region: metadata.region ?? null,
                city: metadata.city ?? null,
                riskScore: risk.riskScore,
                riskReason: risk.riskReason ?? null,
                createdAt: now,
            },
            now,
            transaction,
        );

        return {
            challenge,
            checkpointToken,
            code,
        };
    }

    async getChallenge(checkpointTokenOrId: string): Promise<LoginChallengeSelect | null> {
        let challenge = await this.loginChallengeRepository.findById(checkpointTokenOrId);
        if (!challenge) {
            const tokenHash = this.hash(checkpointTokenOrId);
            challenge = await this.loginChallengeRepository.findByTokenHash(tokenHash);
        }
        return challenge;
    }

    async getLatestChallengeForDevice(
        userId: string,
        realm: AuthRealm,
        deviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect | null> {
        return await this.loginChallengeRepository.findLatestByUserRealmDevice(
            userId,
            realm,
            deviceId,
            transaction,
        );
    }

    getResendEligibility(
        challenge: LoginChallengeSelect,
        now = new Date(),
        latestChallenge?: LoginChallengeSelect | null,
    ): LoginChallengeResendEligibility {
        const resendWindowExpiresAt = new Date(
            challenge.createdAt.getTime() + LOGIN_CHALLENGE_RESEND_WINDOW_MS,
        );

        if (latestChallenge && latestChallenge.id !== challenge.id) {
            return {
                eligible: false,
                reason: 'superseded',
                resendWindowExpiresAt,
            };
        }

        if (challenge.consumedAt !== null) {
            return {
                eligible: false,
                reason: 'consumed',
                resendWindowExpiresAt,
            };
        }

        if (challenge.failedAt !== null || challenge.attemptCount >= challenge.maxAttempts) {
            return {
                eligible: false,
                reason: 'failed',
                resendWindowExpiresAt,
            };
        }

        if (resendWindowExpiresAt.getTime() <= now.getTime()) {
            return {
                eligible: false,
                reason: 'outside_resend_window',
                resendWindowExpiresAt,
            };
        }

        const isVerificationCodeExpired =
            challenge.expiredAt !== null || challenge.expiresAt.getTime() <= now.getTime();

        return {
            eligible: true,
            state: isVerificationCodeExpired ? 'expired' : 'active',
            resendWindowExpiresAt,
        };
    }

    async verifyLoginChallenge(
        checkpointTokenOrId: string,
        code: string,
        now = new Date(),
    ): Promise<boolean> {
        // Look up by ID first, then by token hash if not found
        let challenge = await this.loginChallengeRepository.findById(checkpointTokenOrId);
        if (!challenge) {
            const tokenHash = this.hash(checkpointTokenOrId);
            challenge = await this.loginChallengeRepository.findByTokenHash(tokenHash);
        }

        if (!challenge) {
            return false;
        }

        // Check if expired and update DB if not already done
        const isPastExpiry = challenge.expiresAt.getTime() <= now.getTime();
        if (isPastExpiry && challenge.expiredAt === null) {
            await this.loginChallengeRepository.expire(challenge.id, now);
            return false;
        }

        // Predicates: failedAt is null, expiredAt is null, consumedAt is null, expiresAt > now, attemptCount < maxAttempts
        const isActive =
            challenge.consumedAt === null &&
            challenge.failedAt === null &&
            challenge.expiredAt === null &&
            !isPastExpiry &&
            challenge.attemptCount < challenge.maxAttempts;

        if (!isActive) {
            return false;
        }

        const hashedInputCode = this.hash(code);
        if (challenge.codeHash !== hashedInputCode) {
            // Increment attempt count
            await this.loginChallengeRepository.incrementAttempts(challenge.id, now);
            return false;
        }

        return true;
    }

    async consumeLoginChallenge(challengeId: string, now = new Date()): Promise<boolean> {
        return await this.loginChallengeRepository.consume(challengeId, now);
    }

    async failLoginChallenge(challengeId: string, now = new Date()): Promise<boolean> {
        return await this.loginChallengeRepository.fail(challengeId, now);
    }

    async expireLoginChallenge(challengeId: string, now = new Date()): Promise<boolean> {
        return await this.loginChallengeRepository.expire(challengeId, now);
    }

    async approveAndConsumeLoginChallengeAtomically<T>(
        challengeId: string,
        issueAuthState: (transaction: RepositoryTransaction) => Promise<T>,
        now = new Date(),
    ): Promise<T | null> {
        return await this.loginChallengeRepository.approveAndConsumeAtomically(
            challengeId,
            issueAuthState,
            now,
        );
    }
}
