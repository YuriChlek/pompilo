import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { and, eq, gt, isNull, lt, lte, sql, or, inArray, asc, desc } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    loginChallenges,
    LoginChallengeInsert,
    LoginChallengeSelect,
} from '@/module-auth-token/schemas/login-challenges.schema';
import {
    createRepositoryTransaction,
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';

@Injectable()
export class LoginChallengeRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async createSafe(
        data: LoginChallengeInsert,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const runWithClient = async (client: NodePgDatabase<typeof schema>) => {
            const lockKey = `${data.userId}:${data.realm}:${data.deviceId}`;

            await client.execute(
                sql`select pg_advisory_xact_lock(hashtextextended(${lockKey}, 0))`,
            );

            // Only the latest pending challenge for a device should be usable.
            await client
                .update(loginChallenges)
                .set({ expiredAt: now })
                .where(
                    and(
                        eq(loginChallenges.userId, data.userId),
                        eq(loginChallenges.realm, data.realm),
                        eq(loginChallenges.deviceId, data.deviceId),
                        isNull(loginChallenges.consumedAt),
                        isNull(loginChallenges.failedAt),
                        isNull(loginChallenges.expiredAt),
                    ),
                );

            const [created] = await client.insert(loginChallenges).values(data).returning();
            return created;
        };

        if (transaction) {
            return await runWithClient(dbClient);
        }
        return await this.db.transaction(async tx => {
            return await runWithClient(tx as unknown as NodePgDatabase<typeof schema>);
        });
    }

    async findById(
        id: string,
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [challenge] = await dbClient
            .select()
            .from(loginChallenges)
            .where(eq(loginChallenges.id, id))
            .limit(1);
        return challenge || null;
    }

    async findByTokenHash(
        tokenHash: string,
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [challenge] = await dbClient
            .select()
            .from(loginChallenges)
            .where(eq(loginChallenges.checkpointTokenHash, tokenHash))
            .limit(1);
        return challenge || null;
    }

    async findLatestByUserRealmDevice(
        userId: string,
        realm: AuthRealm,
        deviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [challenge] = await dbClient
            .select()
            .from(loginChallenges)
            .where(
                and(
                    eq(loginChallenges.userId, userId),
                    eq(loginChallenges.realm, realm),
                    eq(loginChallenges.deviceId, deviceId),
                ),
            )
            .orderBy(desc(loginChallenges.createdAt))
            .limit(1);
        return challenge || null;
    }

    async consume(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(loginChallenges)
            .set({ consumedAt: now })
            .where(
                and(
                    eq(loginChallenges.id, id),
                    isNull(loginChallenges.consumedAt),
                    isNull(loginChallenges.failedAt),
                    isNull(loginChallenges.expiredAt),
                    gt(loginChallenges.expiresAt, now),
                    lt(loginChallenges.attemptCount, loginChallenges.maxAttempts),
                ),
            )
            .returning();
        return !!updated;
    }

    async approveAndConsume(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(loginChallenges)
            .set({ approvedAt: now, consumedAt: now })
            .where(
                and(
                    eq(loginChallenges.id, id),
                    isNull(loginChallenges.consumedAt),
                    isNull(loginChallenges.failedAt),
                    isNull(loginChallenges.expiredAt),
                    gt(loginChallenges.expiresAt, now),
                    lt(loginChallenges.attemptCount, loginChallenges.maxAttempts),
                ),
            )
            .returning();
        return !!updated;
    }

    async fail(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(loginChallenges)
            .set({ failedAt: now })
            .where(
                and(
                    eq(loginChallenges.id, id),
                    isNull(loginChallenges.consumedAt),
                    isNull(loginChallenges.failedAt),
                    isNull(loginChallenges.expiredAt),
                    gt(loginChallenges.expiresAt, now),
                    lt(loginChallenges.attemptCount, loginChallenges.maxAttempts),
                ),
            )
            .returning();
        return !!updated;
    }

    async incrementAttempts(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<LoginChallengeSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(loginChallenges)
            .set({
                attemptCount: sql`${loginChallenges.attemptCount} + 1`,
                failedAt: sql`CASE WHEN ${loginChallenges.attemptCount} + 1 >= ${loginChallenges.maxAttempts} THEN ${now.toISOString()}::timestamptz ELSE ${loginChallenges.failedAt} END`,
            })
            .where(
                and(
                    eq(loginChallenges.id, id),
                    isNull(loginChallenges.consumedAt),
                    isNull(loginChallenges.failedAt),
                    isNull(loginChallenges.expiredAt),
                    gt(loginChallenges.expiresAt, now),
                    lt(loginChallenges.attemptCount, loginChallenges.maxAttempts),
                ),
            )
            .returning();
        return updated || null;
    }

    async expire(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(loginChallenges)
            .set({ expiredAt: now })
            .where(
                and(
                    eq(loginChallenges.id, id),
                    isNull(loginChallenges.consumedAt),
                    isNull(loginChallenges.failedAt),
                    isNull(loginChallenges.expiredAt),
                    lte(loginChallenges.expiresAt, now),
                ),
            )
            .returning();
        return !!updated;
    }

    async approveAndConsumeAtomically<T>(
        id: string,
        issueAuthState: (transaction: RepositoryTransaction) => Promise<T>,
        now = new Date(),
    ): Promise<T | null> {
        return this.db.transaction(async tx => {
            const transaction = createRepositoryTransaction(tx as NodePgDatabase<typeof schema>);
            const approved = await this.approveAndConsume(id, now, transaction);

            if (!approved) {
                return null;
            }

            return issueAuthState(transaction);
        });
    }

    async deleteExpiredChallenges(
        retentionPeriodMs = 0,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        const eligibleSubquery = dbClient
            .select({ id: loginChallenges.id })
            .from(loginChallenges)
            .where(
                or(
                    lt(loginChallenges.expiresAt, retentionThreshold),
                    lt(loginChallenges.consumedAt, retentionThreshold),
                    lt(loginChallenges.failedAt, retentionThreshold),
                    lt(loginChallenges.expiredAt, retentionThreshold),
                ),
            )
            .orderBy(asc(loginChallenges.createdAt))
            .limit(1000);

        const result = await dbClient
            .delete(loginChallenges)
            .where(inArray(loginChallenges.id, eligibleSubquery));

        return result.rowCount ?? 0;
    }
}
