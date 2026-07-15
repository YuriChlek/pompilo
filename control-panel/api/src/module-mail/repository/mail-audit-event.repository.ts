import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import { mailAuditEvents } from '@/module-mail/schemas/mail-audit-events.schema';
import type { CreateMailAuditEventInput } from '@/module-mail/interfaces/mail-audit-event.interfaces';

@Injectable()
export class MailAuditEventRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async create(
        input: CreateMailAuditEventInput,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const db = getTransactionClient(transaction, this.db);
        await db.insert(mailAuditEvents).values({
            action: input.action,
            adminUserId: input.adminUserId,
            payload: input.payload,
        });
    }
}
