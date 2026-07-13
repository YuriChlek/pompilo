import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { and, eq, gt, isNull, lt, isNotNull, inArray, asc, or } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    emailVerifications,
    EmailVerificationInsert,
    EmailVerificationSelect,
} from '@/module-auth-token/schemas/email-verifications.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class EmailVerificationRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(
        data: EmailVerificationInsert,
        transaction?: RepositoryTransaction,
    ): Promise<EmailVerificationSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [verification] = await dbClient.insert(emailVerifications).values(data).returning();
        return verification;
    }

    async findByTokenHash(
        hash: string,
        transaction?: RepositoryTransaction,
    ): Promise<EmailVerificationSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [verification] = await dbClient
            .select()
            .from(emailVerifications)
            .where(eq(emailVerifications.tokenHash, hash))
            .limit(1);
        return verification || null;
    }

    async findActiveByUserId(
        userId: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<EmailVerificationSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [verification] = await dbClient
            .select()
            .from(emailVerifications)
            .where(
                and(
                    eq(emailVerifications.userId, userId),
                    isNull(emailVerifications.consumedAt),
                    gt(emailVerifications.expiresAt, now),
                ),
            )
            .limit(1);
        return verification || null;
    }

    async consume(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(emailVerifications)
            .set({ consumedAt: now })
            .where(
                and(
                    eq(emailVerifications.id, id),
                    isNull(emailVerifications.consumedAt),
                    gt(emailVerifications.expiresAt, now),
                ),
            )
            .returning();
        return !!updated;
    }

    async deleteExpired(
        retentionPeriodMs = 0,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        const eligibleSubquery = dbClient
            .select({ id: emailVerifications.id })
            .from(emailVerifications)
            .where(
                or(
                    lt(emailVerifications.expiresAt, retentionThreshold),
                    and(
                        isNotNull(emailVerifications.consumedAt),
                        lt(emailVerifications.consumedAt, retentionThreshold),
                    ),
                ),
            )
            .orderBy(asc(emailVerifications.createdAt))
            .limit(1000);

        const deleted = await dbClient
            .delete(emailVerifications)
            .where(inArray(emailVerifications.id, eligibleSubquery));

        return deleted.rowCount ?? 0;
    }
}
