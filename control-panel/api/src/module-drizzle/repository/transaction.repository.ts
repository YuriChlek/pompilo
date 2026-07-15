import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';

const TRANSACTION_CLIENT = Symbol('TRANSACTION_CLIENT');

type DatabaseClient = NodePgDatabase<typeof schema>;

export interface RepositoryTransaction {
    readonly [TRANSACTION_CLIENT]: DatabaseClient;
}

export function createRepositoryTransaction(client: DatabaseClient): RepositoryTransaction {
    return {
        [TRANSACTION_CLIENT]: client,
    };
}

export function getTransactionClient(
    transaction: RepositoryTransaction | undefined,
    fallback: DatabaseClient,
): DatabaseClient {
    return transaction?.[TRANSACTION_CLIENT] ?? fallback;
}

@Injectable()
export class TransactionRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: DatabaseClient,
    ) {}

    async run<T>(work: (transaction: RepositoryTransaction) => Promise<T>): Promise<T> {
        return this.db.transaction(async transaction =>
            work(createRepositoryTransaction(transaction as unknown as DatabaseClient)),
        );
    }
}
