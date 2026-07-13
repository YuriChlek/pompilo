import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { eq, lt, inArray, asc } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    securityEvents,
    SecurityEventSelect,
} from '@/module-auth-token/schemas/security-events.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import type { SecurityEventWriteInput } from '@/module-auth-token/interfaces/security-event.interfaces';
import { SECURITY_EVENT_CONTRACTS } from '@/module-auth-token/constants/security-event.constants';

@Injectable()
export class SecurityEventRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(
        data: SecurityEventWriteInput,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect> {
        validateSecurityEventInput(data);
        const dbClient = getTransactionClient(transaction, this.db);
        const [event] = await dbClient.insert(securityEvents).values(data).returning();
        return event;
    }

    async findById(
        id: string,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [event] = await dbClient
            .select()
            .from(securityEvents)
            .where(eq(securityEvents.id, id))
            .limit(1);
        return event || null;
    }

    async findByUserId(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return dbClient
            .select()
            .from(securityEvents)
            .where(eq(securityEvents.userId, userId))
            .orderBy(securityEvents.createdAt);
    }

    async deleteExpiredEvents(
        retentionPeriodMs = 0,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        const eligibleSubquery = dbClient
            .select({ id: securityEvents.id })
            .from(securityEvents)
            .where(lt(securityEvents.createdAt, retentionThreshold))
            .orderBy(asc(securityEvents.createdAt))
            .limit(1000);

        const result = await dbClient
            .delete(securityEvents)
            .where(inArray(securityEvents.id, eligibleSubquery));

        return result.rowCount ?? 0;
    }
}

function validateSecurityEventInput(data: SecurityEventWriteInput): void {
    const contract = SECURITY_EVENT_CONTRACTS[data.eventType];
    const input = data as SecurityEventWriteInput & Record<string, unknown>;

    for (const field of contract.requiredFields) {
        if (input[field] === undefined || input[field] === null || input[field] === '') {
            throw new Error(`${data.eventType} requires security-event field "${field}"`);
        }
    }

    if (contract.requiredMetadataFields.length === 0) {
        return;
    }

    if (!isRecord(data.metadata)) {
        throw new Error(`${data.eventType} requires security-event metadata`);
    }

    for (const field of contract.requiredMetadataFields) {
        if (
            data.metadata[field] === undefined ||
            data.metadata[field] === null ||
            data.metadata[field] === ''
        ) {
            throw new Error(`${data.eventType} requires metadata field "${field}"`);
        }
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
