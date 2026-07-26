import { Inject, Injectable } from '@nestjs/common';
import { and, eq, gt, inArray, isNull, lt, or, sql } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import {
    emailChangeChallenges,
    EmailChangeChallengeInsert,
    EmailChangeChallengeSelect,
} from '@/module-account/schemas';

@Injectable()
export class EmailChangeChallengeRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async create(
        data: EmailChangeChallengeInsert,
        transaction?: RepositoryTransaction,
    ): Promise<EmailChangeChallengeSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [created] = await dbClient.insert(emailChangeChallenges).values(data).returning();
        return created;
    }

    async findActiveByUserId(userId: string): Promise<EmailChangeChallengeSelect | null> {
        const now = new Date();
        const [challenge] = await this.db
            .select()
            .from(emailChangeChallenges)
            .where(
                and(
                    eq(emailChangeChallenges.userId, userId),
                    isNull(emailChangeChallenges.usedAt),
                    isNull(emailChangeChallenges.invalidatedAt),
                    gt(emailChangeChallenges.expiresAt, now),
                    or(
                        isNull(emailChangeChallenges.lockedUntil),
                        lt(emailChangeChallenges.lockedUntil, now),
                    ),
                ),
            )
            .limit(1);
        return challenge ?? null;
    }

    async invalidateUserChallenges(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const dbClient = getTransactionClient(transaction, this.db);
        await dbClient
            .update(emailChangeChallenges)
            .set({ invalidatedAt: new Date() })
            .where(
                and(
                    eq(emailChangeChallenges.userId, userId),
                    isNull(emailChangeChallenges.usedAt),
                    isNull(emailChangeChallenges.invalidatedAt),
                ),
            );
    }

    async consume(id: string, transaction?: RepositoryTransaction): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(emailChangeChallenges)
            .set({ usedAt: new Date() })
            .where(
                and(
                    eq(emailChangeChallenges.id, id),
                    isNull(emailChangeChallenges.usedAt),
                    isNull(emailChangeChallenges.invalidatedAt),
                    gt(emailChangeChallenges.expiresAt, new Date()),
                ),
            )
            .returning();
        return !!updated;
    }

    async update(
        id: string,
        data: Partial<EmailChangeChallengeInsert>,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const dbClient = getTransactionClient(transaction, this.db);
        await dbClient
            .update(emailChangeChallenges)
            .set(data)
            .where(eq(emailChangeChallenges.id, id));
    }

    async cleanup(olderThan: Date, transaction?: RepositoryTransaction): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();

        // Cleanup criteria: expired OR (used/invalidated and older than retention period)
        const subquery = dbClient
            .select({ id: emailChangeChallenges.id })
            .from(emailChangeChallenges)
            .where(
                or(
                    lt(emailChangeChallenges.expiresAt, now),
                    lt(emailChangeChallenges.usedAt, olderThan),
                    lt(emailChangeChallenges.invalidatedAt, olderThan),
                ),
            )
            .limit(1000);

        const deleted = await dbClient
            .delete(emailChangeChallenges)
            .where(inArray(emailChangeChallenges.id, subquery))
            .returning({ id: emailChangeChallenges.id });

        return deleted.length;
    }

    async incrementAttempts(
        id: string,
        lockThreshold = 5,
        lockDurationMinutes = 30,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const lockTime = new Date(now.getTime() + lockDurationMinutes * 60 * 1000);

        const [updated] = await dbClient
            .update(emailChangeChallenges)
            .set({
                attemptCount: sql`${emailChangeChallenges.attemptCount} + 1`,
                lockedUntil: sql`CASE WHEN ${emailChangeChallenges.attemptCount} + 1 >= ${lockThreshold} THEN ${lockTime} ELSE ${emailChangeChallenges.lockedUntil} END`,
            })
            .where(eq(emailChangeChallenges.id, id))
            .returning({ attemptCount: emailChangeChallenges.attemptCount });

        return updated ? updated.attemptCount : 0;
    }
}
