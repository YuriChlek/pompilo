import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { and, eq, gt, isNull, ne, or, lt, isNotNull, inArray, asc } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    sessions,
    SessionInsert,
    SessionSelect,
} from '@/module-auth-token/schemas/sessions.schema';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class SessionRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(data: SessionInsert, transaction?: RepositoryTransaction): Promise<SessionSelect> {
        const dbClient = getTransactionClient(transaction, this.db);

        // Validation: sessions.device_id must match known_devices.device_id
        const [knownDevice] = await dbClient
            .select()
            .from(knownDevices)
            .where(eq(knownDevices.id, data.knownDeviceId))
            .limit(1);

        if (!knownDevice) {
            throw new Error(`Known device ${data.knownDeviceId} not found`);
        }

        if (knownDevice.deviceId !== data.deviceId) {
            throw new Error(
                `session.device_id (${data.deviceId}) does not match known_device.device_id (${knownDevice.deviceId})`,
            );
        }

        const [session] = await dbClient.insert(sessions).values(data).returning();
        return session;
    }

    async findById(id: string, transaction?: RepositoryTransaction): Promise<SessionSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [session] = await dbClient
            .select()
            .from(sessions)
            .where(eq(sessions.id, id))
            .limit(1);
        return session || null;
    }

    async findReusable(
        userId: string,
        realm: string,
        deviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [session] = await dbClient
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    eq(sessions.deviceId, deviceId),
                    isNull(sessions.revokedAt),
                ),
            )
            .limit(1);
        return session || null;
    }

    async findActiveByUserRealm(
        userId: string,
        realm: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return dbClient
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    isNull(sessions.revokedAt),
                    gt(sessions.expiresAt, new Date()),
                ),
            );
    }

    async findUnrevokedByUserRealm(
        userId: string,
        realm: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return dbClient
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    isNull(sessions.revokedAt),
                ),
            );
    }

    async findActiveByUser(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return dbClient
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    isNull(sessions.revokedAt),
                    gt(sessions.expiresAt, new Date()),
                ),
            );
    }

    async findUnrevokedOtherByUserRealm(
        userId: string,
        realm: string,
        currentSessionId: string,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return dbClient
            .select()
            .from(sessions)
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    ne(sessions.id, currentSessionId),
                    isNull(sessions.revokedAt),
                ),
            );
    }

    async findUnrevokedByKnownDeviceForUpdate(
        knownDeviceId: string,
        transaction: RepositoryTransaction,
    ): Promise<SessionSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return await dbClient
            .select()
            .from(sessions)
            .where(and(eq(sessions.knownDeviceId, knownDeviceId), isNull(sessions.revokedAt)))
            .for('update');
    }

    async revoke(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(sessions)
            .set({ revokedAt: now, updatedAt: now })
            .where(and(eq(sessions.id, id), isNull(sessions.revokedAt)))
            .returning();
        return !!updated;
    }

    async reuseSession(
        id: string,
        expiresAt: Date,
        metadata?: Partial<SessionInsert>,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(sessions)
            .set({
                ...metadata,
                expiresAt,
                lastSeenAt: now,
                updatedAt: now,
            })
            .where(and(eq(sessions.id, id), isNull(sessions.revokedAt)))
            .returning();
        return updated || null;
    }

    async update(
        id: string,
        data: Partial<SessionInsert>,
        transaction?: RepositoryTransaction,
    ): Promise<SessionSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(sessions)
            .set({
                ...data,
                updatedAt: new Date(),
            })
            .where(eq(sessions.id, id))
            .returning();
        return updated || null;
    }

    async revokeOther(
        userId: string,
        realm: string,
        currentSessionId: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const result = await dbClient
            .update(sessions)
            .set({ revokedAt: now, updatedAt: now })
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    ne(sessions.id, currentSessionId),
                    isNull(sessions.revokedAt),
                ),
            )
            .returning();
        return result.length;
    }

    async revokeAll(
        userId: string,
        realm: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const result = await dbClient
            .update(sessions)
            .set({ revokedAt: now, updatedAt: now })
            .where(
                and(
                    eq(sessions.userId, userId),
                    eq(sessions.realm, realm),
                    isNull(sessions.revokedAt),
                ),
            )
            .returning();
        return result.length;
    }

    async revokeUnrevokedByKnownDevice(
        knownDeviceId: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const result = await dbClient
            .update(sessions)
            .set({ revokedAt: now, updatedAt: now })
            .where(and(eq(sessions.knownDeviceId, knownDeviceId), isNull(sessions.revokedAt)))
            .returning();
        return result.length;
    }

    async deleteExpiredSessions(
        retentionPeriodMs = 0,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        const eligibleSessionsSubquery = dbClient
            .select({ id: sessions.id })
            .from(sessions)
            .where(
                or(
                    and(isNull(sessions.revokedAt), lt(sessions.expiresAt, retentionThreshold)),
                    and(isNotNull(sessions.revokedAt), lt(sessions.revokedAt, retentionThreshold)),
                ),
            )
            .orderBy(asc(sessions.createdAt))
            .limit(1000);

        const result = await dbClient
            .delete(sessions)
            .where(inArray(sessions.id, eligibleSessionsSubquery));

        return result.rowCount ?? 0;
    }
}
