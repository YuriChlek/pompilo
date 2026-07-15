import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { and, eq, gt, isNull, or, lt, isNotNull, inArray, asc } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    reauthConfirmations,
    ReauthConfirmationInsert,
    ReauthConfirmationSelect,
} from '@/module-auth-token/schemas/reauth-confirmations.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';

@Injectable()
export class ReauthConfirmationRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(
        data: ReauthConfirmationInsert,
        transaction?: RepositoryTransaction,
    ): Promise<ReauthConfirmationSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [confirmation] = await dbClient.insert(reauthConfirmations).values(data).returning();
        return confirmation;
    }

    async findById(
        id: string,
        transaction?: RepositoryTransaction,
    ): Promise<ReauthConfirmationSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [confirmation] = await dbClient
            .select()
            .from(reauthConfirmations)
            .where(eq(reauthConfirmations.id, id))
            .limit(1);
        return confirmation || null;
    }

    async findByTokenHash(
        hash: string,
        transaction?: RepositoryTransaction,
    ): Promise<ReauthConfirmationSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [confirmation] = await dbClient
            .select()
            .from(reauthConfirmations)
            .where(eq(reauthConfirmations.confirmationTokenHash, hash))
            .limit(1);
        return confirmation || null;
    }

    async consume(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(reauthConfirmations)
            .set({ consumedAt: now })
            .where(
                and(
                    eq(reauthConfirmations.id, id),
                    isNull(reauthConfirmations.consumedAt),
                    gt(reauthConfirmations.expiresAt, now),
                ),
            )
            .returning();
        return !!updated;
    }

    async consumeByTokenHash(
        hash: string,
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        actionScope: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(reauthConfirmations)
            .set({ consumedAt: now })
            .where(
                and(
                    eq(reauthConfirmations.confirmationTokenHash, hash),
                    eq(reauthConfirmations.userId, userId),
                    eq(reauthConfirmations.realm, realm),
                    eq(reauthConfirmations.sessionId, sessionId),
                    eq(reauthConfirmations.actionScope, actionScope),
                    isNull(reauthConfirmations.consumedAt),
                    gt(reauthConfirmations.expiresAt, now),
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
            .select({ id: reauthConfirmations.id })
            .from(reauthConfirmations)
            .where(
                or(
                    lt(reauthConfirmations.expiresAt, retentionThreshold),
                    and(
                        isNotNull(reauthConfirmations.consumedAt),
                        lt(reauthConfirmations.consumedAt, retentionThreshold),
                    ),
                ),
            )
            .orderBy(asc(reauthConfirmations.createdAt))
            .limit(1000);

        const deleted = await dbClient
            .delete(reauthConfirmations)
            .where(inArray(reauthConfirmations.id, eligibleSubquery));

        return deleted.rowCount ?? 0;
    }
}
