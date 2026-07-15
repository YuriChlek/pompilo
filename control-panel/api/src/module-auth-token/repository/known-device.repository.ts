import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { and, eq, isNull, lt, isNotNull, notInArray, inArray, asc } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    knownDevices,
    KnownDeviceInsert,
    KnownDeviceSelect,
} from '@/module-auth-token/schemas/known-devices.schema';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';
import type { KnownDeviceMetadataUpdate } from '@/module-auth-token/types/known-device.types';

@Injectable()
export class KnownDeviceRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(
        data: KnownDeviceInsert,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [device] = await dbClient.insert(knownDevices).values(data).returning();
        return device;
    }

    async findById(
        id: string,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [device] = await dbClient
            .select()
            .from(knownDevices)
            .where(eq(knownDevices.id, id))
            .limit(1);
        return device || null;
    }

    async findByIdForUpdate(
        id: string,
        transaction: RepositoryTransaction,
    ): Promise<KnownDeviceSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [device] = await dbClient
            .select()
            .from(knownDevices)
            .where(eq(knownDevices.id, id))
            .for('update');
        return device || null;
    }

    async findActiveByDevice(
        userId: string,
        realm: AuthRealm,
        deviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [device] = await dbClient
            .select()
            .from(knownDevices)
            .where(
                and(
                    eq(knownDevices.userId, userId),
                    eq(knownDevices.realm, realm),
                    eq(knownDevices.deviceId, deviceId),
                    isNull(knownDevices.revokedAt),
                ),
            )
            .limit(1);
        return device || null;
    }

    async update(
        id: string,
        data: KnownDeviceMetadataUpdate,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [device] = await dbClient
            .update(knownDevices)
            .set({
                ...data,
                updatedAt: new Date(),
            })
            .where(eq(knownDevices.id, id))
            .returning();
        return device || null;
    }

    async revoke(
        id: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [updated] = await dbClient
            .update(knownDevices)
            .set({
                revokedAt: now,
                trustedAt: null,
                trustExpiresAt: null,
                updatedAt: now,
            })
            .where(and(eq(knownDevices.id, id), isNull(knownDevices.revokedAt)))
            .returning();
        return !!updated;
    }

    async listActive(
        userId: string,
        realm: AuthRealm,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        return await dbClient
            .select()
            .from(knownDevices)
            .where(
                and(
                    eq(knownDevices.userId, userId),
                    eq(knownDevices.realm, realm),
                    isNull(knownDevices.revokedAt),
                ),
            );
    }

    async deleteExpiredKnownDevices(
        retentionPeriodMs = 0,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        // Subquery for all known_device_id values currently referenced in sessions
        const referencedDeviceIds = dbClient
            .select({ knownDeviceId: sessions.knownDeviceId })
            .from(sessions);

        const eligibleDevicesSubquery = dbClient
            .select({ id: knownDevices.id })
            .from(knownDevices)
            .where(
                and(
                    isNotNull(knownDevices.revokedAt),
                    lt(knownDevices.revokedAt, retentionThreshold),
                    notInArray(knownDevices.id, referencedDeviceIds),
                ),
            )
            .orderBy(asc(knownDevices.createdAt))
            .limit(1000);

        const result = await dbClient
            .delete(knownDevices)
            .where(inArray(knownDevices.id, eligibleDevicesSubquery));

        return result.rowCount ?? 0;
    }
}
