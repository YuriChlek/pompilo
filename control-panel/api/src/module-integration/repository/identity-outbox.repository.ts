import { Inject, Injectable } from '@nestjs/common';
import { and, asc, eq, inArray, lte } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import {
    identityOutboxEvents,
    IdentityOutboxEventInsert,
    IdentityOutboxEventSelect,
} from '@/module-integration/schemas';

@Injectable()
export class IdentityOutboxRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async create(
        data: Omit<
            IdentityOutboxEventInsert,
            'id' | 'status' | 'createdAt' | 'updatedAt' | 'publishedAt' | 'lastError'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<IdentityOutboxEventSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [created] = await dbClient
            .insert(identityOutboxEvents)
            .values({
                ...data,
                status: 'pending',
            })
            .onConflictDoNothing()
            .returning();

        return created ?? null;
    }

    async listPending(limit: number, now = new Date()): Promise<IdentityOutboxEventSelect[]> {
        return this.db
            .select()
            .from(identityOutboxEvents)
            .where(
                and(
                    eq(identityOutboxEvents.status, 'pending'),
                    lte(identityOutboxEvents.availableAt, now),
                ),
            )
            .orderBy(asc(identityOutboxEvents.createdAt))
            .limit(limit);
    }

    async markPublished(eventIds: string[], publishedAt = new Date()): Promise<number> {
        if (eventIds.length === 0) {
            return 0;
        }

        const updated = await this.db
            .update(identityOutboxEvents)
            .set({
                status: 'published',
                publishedAt,
                updatedAt: publishedAt,
            })
            .where(
                and(
                    inArray(identityOutboxEvents.eventId, eventIds),
                    eq(identityOutboxEvents.status, 'pending'),
                ),
            )
            .returning({ id: identityOutboxEvents.id });

        return updated.length;
    }
}
