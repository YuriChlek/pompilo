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
    passwordResetChallenges,
    PasswordResetChallengeInsert,
    PasswordResetChallengeSelect,
} from '@/module-account/schemas';

@Injectable()
export class PasswordResetChallengeRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async create(
        data: PasswordResetChallengeInsert,
        transaction?: RepositoryTransaction,
    ): Promise<PasswordResetChallengeSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [created] = await dbClient.insert(passwordResetChallenges).values(data).returning();
        return created;
    }

    async findBySelector(selector: string): Promise<PasswordResetChallengeSelect | null> {
        const [challenge] = await this.db
            .select()
            .from(passwordResetChallenges)
            .where(eq(passwordResetChallenges.selector, selector))
            .limit(1);
        return challenge ?? null;
    }

    async findActiveBySelector(selector: string): Promise<PasswordResetChallengeSelect | null> {
        const now = new Date();
        const [challenge] = await this.db
            .select()
            .from(passwordResetChallenges)
            .where(
                and(
                    eq(passwordResetChallenges.selector, selector),
                    isNull(passwordResetChallenges.usedAt),
                    isNull(passwordResetChallenges.invalidatedAt),
                    gt(passwordResetChallenges.expiresAt, now),
                    or(
                        isNull(passwordResetChallenges.lockedUntil),
                        lt(passwordResetChallenges.lockedUntil, now),
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
            .update(passwordResetChallenges)
            .set({ invalidatedAt: new Date() })
            .where(
                and(
                    eq(passwordResetChallenges.userId, userId),
                    isNull(passwordResetChallenges.usedAt),
                    isNull(passwordResetChallenges.invalidatedAt),
                ),
            );
    }

    async consume(id: string, transaction?: RepositoryTransaction): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(passwordResetChallenges)
            .set({ usedAt: new Date() })
            .where(
                and(
                    eq(passwordResetChallenges.id, id),
                    isNull(passwordResetChallenges.usedAt),
                    isNull(passwordResetChallenges.invalidatedAt),
                    gt(passwordResetChallenges.expiresAt, new Date()),
                ),
            )
            .returning();
        return !!updated;
    }

    async update(
        id: string,
        data: Partial<PasswordResetChallengeInsert>,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const dbClient = getTransactionClient(transaction, this.db);
        await dbClient
            .update(passwordResetChallenges)
            .set(data)
            .where(eq(passwordResetChallenges.id, id));
    }

    async cleanup(olderThan: Date, transaction?: RepositoryTransaction): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();

        // Cleanup criteria: expired OR (used/invalidated and older than retention period)
        const subquery = dbClient
            .select({ id: passwordResetChallenges.id })
            .from(passwordResetChallenges)
            .where(
                or(
                    lt(passwordResetChallenges.expiresAt, now),
                    lt(passwordResetChallenges.usedAt, olderThan),
                    lt(passwordResetChallenges.invalidatedAt, olderThan),
                ),
            )
            .limit(1000);

        const deleted = await dbClient
            .delete(passwordResetChallenges)
            .where(inArray(passwordResetChallenges.id, subquery))
            .returning({ id: passwordResetChallenges.id });

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
            .update(passwordResetChallenges)
            .set({
                attemptCount: sql`${passwordResetChallenges.attemptCount} + 1`,
                lockedUntil: sql`CASE WHEN ${passwordResetChallenges.attemptCount} + 1 >= ${lockThreshold} THEN ${lockTime} ELSE ${passwordResetChallenges.lockedUntil} END`,
            })
            .where(eq(passwordResetChallenges.id, id))
            .returning({ attemptCount: passwordResetChallenges.attemptCount });

        return updated ? updated.attemptCount : 0;
    }
}
