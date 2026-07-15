import {
    Injectable,
    NotFoundException,
    ForbiddenException,
    BadRequestException,
} from '@nestjs/common';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { ConfigService } from '@nestjs/config';
import ms, { type StringValue } from 'ms';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { parseUserAgent } from '@/common/utils/request-metadata.util';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';
import {
    RevokeCurrentSessionInput,
    RevokeOtherSessionsInput,
    RevokeSpecificSessionInput,
    RevokeUserSessionsInput,
    RevokeAllSessionsInput,
    SessionMetadata,
} from '@/module-auth-token/types/session.types';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { UserRepository } from '@/module-user/repository/user.repository';

interface DatabaseError extends Error {
    code?: string;
}

@Injectable()
export class SessionService {
    public constructor(
        private readonly sessionRepository: SessionRepository,
        private readonly knownDeviceRepository: KnownDeviceRepository,
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly securityEventService: SecurityEventService,
        private readonly transactionRepository: TransactionRepository,
        private readonly redisTokenService: RedisTokenService,
        private readonly configService: ConfigService,
        private readonly reauthConfirmationService: ReauthConfirmationService,
        private readonly userRepository: UserRepository,
        private readonly mailTemplateService: MailTemplateService,
    ) {}

    async findReusableSession(
        userId: string,
        realm: string,
        knownDeviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect | null> {
        const device = await this.knownDeviceRepository.findById(knownDeviceId, transaction);
        if (!device || device.userId !== userId || device.realm !== realm || device.revokedAt) {
            return null;
        }

        return await this.sessionRepository.findReusable(
            userId,
            realm,
            device.deviceId,
            transaction,
        );
    }

    async createSession(
        userId: string,
        realm: string,
        knownDeviceId: string,
        deviceId: string,
        metadata?: SessionMetadata,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect> {
        const sessionMaxTtlStr = this.configService.getOrThrow<StringValue>('SESSION_MAX_TTL');
        const expiresAt = new Date(Date.now() + ms(sessionMaxTtlStr));

        try {
            return await this.sessionRepository.save(
                {
                    userId,
                    realm,
                    knownDeviceId,
                    deviceId,
                    expiresAt,
                    ...metadata,
                },
                transaction,
            );
        } catch (error) {
            // Check if it's a unique constraint error (Postgres code 23505)
            const isUniqueViolation =
                error instanceof Error && (error as DatabaseError).code === '23505';
            if (isUniqueViolation) {
                const existing = await this.sessionRepository.findReusable(
                    userId,
                    realm,
                    deviceId,
                    transaction,
                );
                if (existing) {
                    return await this.reuseSession(existing.id, metadata, expiresAt, transaction);
                }
            }
            throw error;
        }
    }

    async reuseSession(
        sessionId: string,
        metadata?: SessionMetadata,
        expiresAt?: Date,
        transaction?: RepositoryTransaction,
        now = new Date(),
    ): Promise<SessionSelect> {
        const sessionMaxTtlStr = this.configService.getOrThrow<StringValue>('SESSION_MAX_TTL');
        const finalExpiresAt = expiresAt ?? new Date(now.getTime() + ms(sessionMaxTtlStr));

        const updated = await this.sessionRepository.reuseSession(
            sessionId,
            finalExpiresAt,
            metadata,
            now,
            transaction,
        );

        if (!updated) {
            throw new NotFoundException(`Session ${sessionId} not found or already revoked`);
        }

        return updated;
    }

    async getSession(
        sessionId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect | null> {
        return await this.sessionRepository.findById(sessionId, transaction);
    }

    async touchLastSeenAt(
        sessionId: string,
        throttleWindowSeconds: number,
        transaction?: RepositoryTransaction,
        now = new Date(),
    ): Promise<SessionSelect | null> {
        const session = await this.sessionRepository.findById(sessionId, transaction);
        if (!session || session.revokedAt) {
            return null;
        }

        const timeDiffMs = now.getTime() - session.lastSeenAt.getTime();
        if (timeDiffMs < throttleWindowSeconds * 1000) {
            return session;
        }

        return await this.sessionRepository.update(
            sessionId,
            {
                lastSeenAt: now,
            },
            transaction,
        );
    }

    async revokeSession(
        sessionId: string,
        userId?: string,
        realm?: string,
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const session = await this.sessionRepository.findById(sessionId, transaction);
        if (!session) {
            throw new NotFoundException(`Session ${sessionId} not found`);
        }

        if (userId !== undefined && session.userId !== userId) {
            throw new ForbiddenException('Access denied');
        }
        if (realm !== undefined && session.realm !== realm) {
            throw new ForbiddenException('Access denied');
        }

        if (session.revokedAt) {
            return false;
        }

        return await this.sessionRepository.revoke(sessionId, new Date(), transaction);
    }

    async revokeCurrentSession(input: RevokeCurrentSessionInput): Promise<void> {
        const denyListTtlSeconds = this.getAccessDenyListTtlSeconds();

        await this.redisTokenService.revokeSession(input.sessionId, denyListTtlSeconds);
        if (input.accessTokenJti) {
            await this.redisTokenService.revokeToken(input.accessTokenJti, denyListTtlSeconds);
        }

        await this.transactionRepository.run(async transaction => {
            const session = await this.sessionRepository.findById(input.sessionId, transaction);
            if (!session) {
                throw new NotFoundException(`Session ${input.sessionId} not found`);
            }

            if (session.userId !== input.userId || session.realm !== input.realm) {
                throw new ForbiddenException('Access denied');
            }

            const now = new Date();
            if (!session.revokedAt) {
                await this.sessionRepository.revoke(input.sessionId, now, transaction);
            }

            await this.authTokenRepository.revokeTokensBySession(input.sessionId, now, transaction);
            await this.securityEventService.recordLogout(
                {
                    userId: input.userId,
                    realm: input.realm,
                    sessionId: input.sessionId,
                    ipAddress: input.ipAddress ?? session.ipAddress ?? undefined,
                    userAgent: input.userAgent ?? session.userAgent ?? undefined,
                },
                transaction,
            );
        });
    }

    async revokeSpecificSession(input: RevokeSpecificSessionInput): Promise<void> {
        const denyListTtlSeconds = this.getAccessDenyListTtlSeconds();

        await this.redisTokenService.revokeSession(input.sessionId, denyListTtlSeconds);

        await this.transactionRepository.run(async transaction => {
            const session = await this.sessionRepository.findById(input.sessionId, transaction);
            if (!session) {
                throw new NotFoundException(`Session ${input.sessionId} not found`);
            }

            if (session.userId !== input.userId || session.realm !== input.realm) {
                throw new ForbiddenException('Access denied');
            }

            const now = new Date();
            if (!session.revokedAt) {
                await this.sessionRepository.revoke(input.sessionId, now, transaction);
            }

            await this.authTokenRepository.revokeTokensBySession(input.sessionId, now, transaction);
            await this.securityEventService.recordSessionRevoked(
                {
                    userId: input.userId,
                    realm: input.realm,
                    sessionId: input.sessionId,
                    metadata: {
                        revocationReason: 'specific_session_revoke',
                    },
                },
                transaction,
            );

            const user = await this.userRepository.findById(input.userId, transaction);
            if (user && user.email) {
                const uaParsed = parseUserAgent(session.userAgent || '');
                const details = `Session revoked. Device: ${uaParsed.os}, Browser: ${uaParsed.browser}. IP: ${session.ipAddress || 'unknown'}.`;
                await this.mailTemplateService.sendSecurityAlert(
                    user.email,
                    user.name || 'User',
                    'Session Revoked',
                    details,
                    now.toISOString(),
                    transaction,
                );
            }
        });
    }

    async revokeOtherSessions(input: RevokeOtherSessionsInput): Promise<number> {
        const denyListTtlSeconds = this.getAccessDenyListTtlSeconds();

        // 1. Validate ownership of the current session outside the transaction.
        const currentSession = await this.sessionRepository.findById(input.currentSessionId);
        if (!currentSession) {
            throw new NotFoundException(`Session ${input.currentSessionId} not found`);
        }
        if (currentSession.userId !== input.userId || currentSession.realm !== input.realm) {
            throw new ForbiddenException('Access denied');
        }

        // 2. Select all unrevoked other sessions outside the transaction.
        const otherSessions = await this.sessionRepository.findUnrevokedOtherByUserRealm(
            input.userId,
            input.realm,
            input.currentSessionId,
        );

        // 3. Write Redis deny-list keys before opening the DB transaction (fail-closed).
        //    If Redis write fails here, the DB transaction never starts — no partial commit.
        //    If DB commit later fails, the false-positive deny-list keys are safe: they expire by TTL.
        for (const session of otherSessions) {
            await this.redisTokenService.revokeSession(session.id, denyListTtlSeconds);
        }

        // 4. Atomically revoke sessions, tokens and record the security event.
        const now = new Date();
        await this.transactionRepository.run(async transaction => {
            const isEnforcementEnabled = this.configService.get<boolean>(
                'REAUTH_ENFORCEMENT_ENABLED',
                false,
            );
            if (isEnforcementEnabled) {
                if (!input.reauthConfirmationToken) {
                    throw new BadRequestException(
                        'Re-authentication confirmation token is required',
                    );
                }
                const consumed = await this.reauthConfirmationService.consumeReauthConfirmation(
                    input.reauthConfirmationToken,
                    input.userId,
                    input.realm,
                    input.currentSessionId,
                    'revoke_other_sessions',
                    now,
                    transaction,
                );
                if (!consumed) {
                    throw new BadRequestException(
                        'Invalid or expired re-authentication confirmation token',
                    );
                }
            }

            if (otherSessions.length === 0) {
                return;
            }

            for (const session of otherSessions) {
                await this.sessionRepository.revoke(session.id, now, transaction);
                await this.authTokenRepository.revokeTokensBySession(session.id, now, transaction);
            }

            await this.securityEventService.recordRevokeOtherSessions(
                {
                    userId: input.userId,
                    realm: input.realm,
                    sessionId: input.currentSessionId,
                    metadata: {
                        revokedSessionCount: otherSessions.length,
                    },
                },
                transaction,
            );

            const user = await this.userRepository.findById(input.userId, transaction);
            if (user && user.email) {
                const details = `Other active sessions of your account were successfully terminated (Count: ${otherSessions.length}).`;
                await this.mailTemplateService.sendSecurityAlert(
                    user.email,
                    user.name || 'User',
                    'Other Sessions Revoked',
                    details,
                    now.toISOString(),
                    transaction,
                );
            }
        });

        return otherSessions.length;
    }

    async revokeAllSessions(input: RevokeAllSessionsInput): Promise<number> {
        const denyListTtlSeconds = this.getAccessDenyListTtlSeconds();

        // 1. Validate ownership of the current session outside the transaction.
        const currentSession = await this.sessionRepository.findById(input.currentSessionId);
        if (!currentSession) {
            throw new NotFoundException(`Session ${input.currentSessionId} not found`);
        }
        if (currentSession.userId !== input.userId || currentSession.realm !== input.realm) {
            throw new ForbiddenException('Access denied');
        }

        // 2. Select all unrevoked sessions (including current and expired-but-unrevoked) outside the transaction.
        const sessions = await this.sessionRepository.findUnrevokedByUserRealm(
            input.userId,
            input.realm,
        );
        if (sessions.length === 0) {
            return 0;
        }

        // 3. Write Redis deny-list keys before opening the DB transaction (fail-closed).
        for (const session of sessions) {
            await this.redisTokenService.revokeSession(session.id, denyListTtlSeconds);
        }

        // 4. Atomically revoke sessions, tokens and record the security event.
        const now = new Date();
        await this.transactionRepository.run(async transaction => {
            for (const session of sessions) {
                await this.sessionRepository.revoke(session.id, now, transaction);
                await this.authTokenRepository.revokeTokensBySession(session.id, now, transaction);
            }

            await this.securityEventService.recordRevokeAllSessions(
                {
                    userId: input.userId,
                    realm: input.realm,
                    sessionId: input.currentSessionId,
                    metadata: {
                        revokedSessionCount: sessions.length,
                        revocationReason: 'user_requested_all_sessions_revoke',
                    },
                },
                transaction,
            );

            const user = await this.userRepository.findById(input.userId, transaction);
            if (user && user.email) {
                const details = `All active sessions of your account were successfully terminated (Count: ${sessions.length}).`;
                await this.mailTemplateService.sendSecurityAlert(
                    user.email,
                    user.name || 'User',
                    'All Sessions Revoked',
                    details,
                    now.toISOString(),
                    transaction,
                );
            }
        });

        return sessions.length;
    }

    async revokeUserSessions(
        input: RevokeUserSessionsInput,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const denyListTtlSeconds = this.getAccessDenyListTtlSeconds();

        if (transaction) {
            return await this.revokeUserSessionsWithinTransaction(
                input,
                denyListTtlSeconds,
                transaction,
            );
        }

        return await this.transactionRepository.run(async newTransaction =>
            this.revokeUserSessionsWithinTransaction(input, denyListTtlSeconds, newTransaction),
        );
    }

    private async revokeUserSessionsWithinTransaction(
        input: RevokeUserSessionsInput,
        denyListTtlSeconds: number,
        transaction: RepositoryTransaction,
    ): Promise<number> {
        const sessions = await this.sessionRepository.findActiveByUser(input.userId, transaction);

        for (const session of sessions) {
            await this.redisTokenService.revokeSession(session.id, denyListTtlSeconds);
        }

        const now = new Date();
        for (const session of sessions) {
            await this.sessionRepository.revoke(session.id, now, transaction);
            await this.authTokenRepository.revokeTokensBySession(session.id, now, transaction);
        }

        if (input.eventType === SecurityEventType.PASSWORD_RESET_COMPLETED) {
            const byRealm = sessions.reduce<Record<'customer' | 'admin', number>>(
                (accumulator, session) => {
                    const realm = session.realm === 'admin' ? 'admin' : 'customer';
                    accumulator[realm] += 1;
                    return accumulator;
                },
                { customer: 0, admin: 0 },
            );
            if (sessions.length === 0 && input.fallbackRealm) {
                byRealm[input.fallbackRealm] = 0;
            }

            for (const [realm, count] of Object.entries(byRealm) as Array<
                ['customer' | 'admin', number]
            >) {
                if (count === 0 && !(sessions.length === 0 && realm === input.fallbackRealm)) {
                    continue;
                }

                await this.securityEventService.recordPasswordResetCompleted(
                    {
                        userId: input.userId,
                        realm,
                        metadata: {
                            revokedSessionCount: count,
                        },
                    },
                    transaction,
                );
            }
        }

        return sessions.length;
    }

    async listActiveSessions(
        userId: string,
        realm: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        return await this.sessionRepository.findActiveByUserRealm(userId, realm, transaction);
    }

    private getAccessDenyListTtlSeconds(): number {
        const accessTokenTtlSeconds = Math.ceil(
            ms(this.configService.getOrThrow<StringValue>('JWT_ACCESS_TOKEN_TTL')) / 1000,
        );
        const clockSkewSeconds = this.configService.getOrThrow<number>('AUTH_CLOCK_SKEW_SECONDS');

        return accessTokenTtlSeconds + clockSkewSeconds;
    }
}
