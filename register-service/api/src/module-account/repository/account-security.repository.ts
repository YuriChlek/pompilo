import { Inject, Injectable } from '@nestjs/common';
import { and, desc, eq, gt, ne, isNull, sql } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import { knownDevices, sessions } from '@/module-auth-token/schemas';
import { users } from '@/module-user/schemas';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';

@Injectable()
export class AccountSecurityRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async findActiveSessions(userId: string, realm: AuthRealm) {
        return this.db
            .select({
                id: sessions.id,
                ipAddress: sessions.ipAddress,
                userAgent: sessions.userAgent,
                createdAt: sessions.createdAt,
                lastSeenAt: sessions.lastSeenAt,
                trustedAt: knownDevices.trustedAt,
                trustExpiresAt: knownDevices.trustExpiresAt,
                riskScore: sessions.riskScore,
                approximateLocation: sql<string | null>`
                    nullif(
                        concat_ws(
                            ', ',
                            nullif(${sessions.lastCity}, ''),
                            nullif(${sessions.lastRegion}, ''),
                            nullif(${sessions.lastCountry}, '')
                        ),
                        ''
                    )
                `,
            })
            .from(sessions)
            .innerJoin(knownDevices, eq(sessions.knownDeviceId, knownDevices.id))
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    isNull(sessions.revokedAt),
                    gt(sessions.expiresAt, new Date()),
                ),
            )
            .orderBy(desc(sessions.lastSeenAt));
    }

    async findSessionByIdForUser(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        transaction: RepositoryTransaction,
    ) {
        const db = getTransactionClient(transaction, this.db);
        const [session] = await db
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.id, sessionId),
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                ),
            )
            .limit(1);

        return session ?? null;
    }

    async findActiveOtherSessionIds(
        userId: string,
        currentSessionId: string,
        transaction: RepositoryTransaction,
    ): Promise<string[]> {
        const db = getTransactionClient(transaction, this.db);
        const activeSessions = await db
            .select({ id: sessions.id })
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    isNull(sessions.revokedAt),
                    gt(sessions.expiresAt, new Date()),
                    ne(sessions.id, currentSessionId),
                ),
            );

        return activeSessions.map(session => session.id);
    }

    async findActiveSessionIds(
        userId: string,
        transaction: RepositoryTransaction,
    ): Promise<string[]> {
        const db = getTransactionClient(transaction, this.db);
        const activeSessions = await db
            .select({ id: sessions.id })
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    isNull(sessions.revokedAt),
                    gt(sessions.expiresAt, new Date()),
                ),
            );

        return activeSessions.map(session => session.id);
    }

    async updatePassword(
        userId: string,
        password: string,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const db = getTransactionClient(transaction, this.db);
        await db.update(users).set({ password }).where(eq(users.id, userId));
    }

    async updateEmail(
        userId: string,
        email: string,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const db = getTransactionClient(transaction, this.db);
        await db
            .update(users)
            .set({
                email,
                emailVerifiedAt: new Date(),
                pendingEmailChange: null,
            })
            .where(eq(users.id, userId));
    }

    async deactivateUser(userId: string, transaction: RepositoryTransaction): Promise<void> {
        const db = getTransactionClient(transaction, this.db);
        await db
            .update(users)
            .set({ isActive: false, accountStatus: 'DEACTIVATED' })
            .where(eq(users.id, userId));
    }
}
