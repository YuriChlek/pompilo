import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { eq } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import { mailSettings, MailSettingsSelect, MailSettingsInsert } from '@/module-mail/schemas';

@Injectable()
export class MailSettingsRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async findSingleton(transaction?: RepositoryTransaction): Promise<MailSettingsSelect | null> {
        const db = getTransactionClient(transaction, this.db);
        const [settings] = await db
            .select()
            .from(mailSettings)
            .where(eq(mailSettings.singletonKey, true))
            .limit(1);

        return settings || null;
    }

    async create(
        data: MailSettingsInsert,
        transaction?: RepositoryTransaction,
    ): Promise<MailSettingsSelect | null> {
        const db = getTransactionClient(transaction, this.db);
        const [settings] = await db
            .insert(mailSettings)
            .values({
                ...data,
                singletonKey: true,
            })
            .onConflictDoNothing({ target: mailSettings.singletonKey })
            .returning();

        return settings || null;
    }

    async update(
        id: string,
        data: Partial<MailSettingsInsert>,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const db = getTransactionClient(transaction, this.db);
        await db
            .update(mailSettings)
            .set({
                ...data,
                updatedAt: new Date(),
            })
            .where(eq(mailSettings.id, id));
    }
}
