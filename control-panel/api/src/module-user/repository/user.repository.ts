import { Inject, Injectable } from '@nestjs/common';
import { and, eq, lte, ne, or, inArray } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import { UserDeleteResult } from '@/module-user/interfaces/user.interfaces';
import { users, UserInsert, UserSelect } from '@/module-user/schemas';
import { NewUserModel, UserModel, UserUpdateModel } from '@/module-user/types/user.types';
import { tokens } from '@/module-auth-token/schemas/tokens.schema';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class UserRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async save(user: NewUserModel, transaction?: RepositoryTransaction): Promise<UserModel> {
        const dbClient = getTransactionClient(transaction, this.db);
        const insertPayload: UserInsert = {
            name: user.name,
            email: user.email.toLowerCase(),
            password: user.password,
            role: user.role,
            isActive: user.isActive,
            createdAt: user.createdAt,
            updatedAt: user.updatedAt,
        };

        const [createdUser] = await dbClient.insert(users).values(insertPayload).returning();

        return this.mapToUser(createdUser);
    }

    async createWithTenant(
        user: NewUserModel,
        tenantName: string,
        transaction?: RepositoryTransaction,
    ): Promise<{
        user: UserModel;
        tenant: schema.TenantSelect;
        membership: schema.MembershipSelect;
    }> {
        const executeCreate = async (dbClient: NodePgDatabase<typeof schema>) => {
            const [createdUser] = await dbClient
                .insert(users)
                .values({
                    name: user.name,
                    email: user.email.toLowerCase(),
                    password: user.password,
                    role: user.role,
                    isActive: user.isActive,
                })
                .returning();

            const [createdTenant] = await dbClient
                .insert(schema.tenants)
                .values({
                    name: tenantName,
                })
                .returning();

            const [createdMembership] = await dbClient
                .insert(schema.memberships)
                .values({
                    userId: createdUser.id,
                    tenantId: createdTenant.id,
                    role: 'OWNER',
                })
                .returning();

            return {
                user: this.mapToUser(createdUser),
                tenant: createdTenant,
                membership: createdMembership,
            };
        };

        if (transaction) {
            return await executeCreate(getTransactionClient(transaction, this.db));
        }

        return await this.db.transaction(async tx => {
            return await executeCreate(tx as unknown as NodePgDatabase<typeof schema>);
        });
    }

    async findByNameOrEmail(
        name: string,
        email: string,
        transaction?: RepositoryTransaction,
    ): Promise<UserModel[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        const foundUsers = await dbClient
            .select()
            .from(users)
            .where(or(eq(users.name, name), eq(users.email, email.toLowerCase())));

        return foundUsers.map(user => this.mapToUser(user));
    }

    async findById(id: string, transaction?: RepositoryTransaction): Promise<UserModel | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [user] = await dbClient.select().from(users).where(eq(users.id, id)).limit(1);

        return user ? this.mapToUser(user) : null;
    }

    async findPrimaryMembership(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<schema.MembershipSelect | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [membership] = await dbClient
            .select()
            .from(schema.memberships)
            .where(eq(schema.memberships.userId, userId))
            .limit(1);

        return membership ?? null;
    }

    async findByLogin(
        login: string,
        transaction?: RepositoryTransaction,
    ): Promise<UserModel | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const lowerLogin = login.toLowerCase();
        const [user] = await dbClient
            .select()
            .from(users)
            .where(or(eq(users.email, lowerLogin), eq(users.name, login)))
            .limit(1);

        return user ? this.mapToUser(user) : null;
    }

    async findByEmail(
        email: string,
        excludeId?: string,
        transaction?: RepositoryTransaction,
    ): Promise<UserModel | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const lowerEmail = email.toLowerCase();
        const condition = excludeId
            ? and(eq(users.email, lowerEmail), ne(users.id, excludeId))
            : eq(users.email, lowerEmail);
        const [user] = await dbClient.select().from(users).where(condition).limit(1);

        return user ? this.mapToUser(user) : null;
    }

    async update(
        id: string,
        updateData: UserUpdateModel,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const dbClient = getTransactionClient(transaction, this.db);
        const dataToUpdate = { ...updateData };
        if (dataToUpdate.email) {
            dataToUpdate.email = dataToUpdate.email.toLowerCase();
        }
        await dbClient
            .update(users)
            .set({
                ...dataToUpdate,
                updatedAt: new Date(),
            })
            .where(eq(users.id, id));
    }

    async delete(id: string, transaction?: RepositoryTransaction): Promise<UserDeleteResult> {
        const dbClient = getTransactionClient(transaction, this.db);

        const executeDelete = async (client: NodePgDatabase<typeof schema>) => {
            // 1. Delete dependent tokens
            await client
                .delete(tokens)
                .where(
                    inArray(
                        tokens.sessionId,
                        client
                            .select({ id: sessions.id })
                            .from(sessions)
                            .where(eq(sessions.userId, id)),
                    ),
                );

            // 2. Delete dependent sessions
            await client.delete(sessions).where(eq(sessions.userId, id));

            // 3. Delete dependent known devices
            await client.delete(knownDevices).where(eq(knownDevices.userId, id));

            // 4. Delete the user
            return client.delete(users).where(eq(users.id, id)).returning({ id: users.id });
        };

        const deletedUsers = transaction
            ? await executeDelete(dbClient)
            : await this.db.transaction(async tx =>
                  executeDelete(tx as unknown as NodePgDatabase<typeof schema>),
              );

        return {
            raw: deletedUsers,
            affected: deletedUsers.length,
        };
    }

    async deleteExpiredUsers(
        now: Date = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);

        const executeDeleteExpired = async (client: NodePgDatabase<typeof schema>) => {
            const expiredUsers = await client
                .select({ id: users.id })
                .from(users)
                .where(lte(users.deletionScheduledAt, now));

            if (expiredUsers.length === 0) {
                return 0;
            }

            const expiredUserIds = expiredUsers.map(u => u.id);

            // 1. Delete dependent tokens
            await client
                .delete(tokens)
                .where(
                    inArray(
                        tokens.sessionId,
                        client
                            .select({ id: sessions.id })
                            .from(sessions)
                            .where(inArray(sessions.userId, expiredUserIds)),
                    ),
                );

            // 2. Delete dependent sessions
            await client.delete(sessions).where(inArray(sessions.userId, expiredUserIds));

            // 3. Delete dependent known devices
            await client.delete(knownDevices).where(inArray(knownDevices.userId, expiredUserIds));

            // 4. Delete the users
            const deleted = await client
                .delete(users)
                .where(inArray(users.id, expiredUserIds))
                .returning({ id: users.id });

            return deleted.length;
        };

        return transaction
            ? await executeDeleteExpired(dbClient)
            : await this.db.transaction(async tx =>
                  executeDeleteExpired(tx as unknown as NodePgDatabase<typeof schema>),
              );
    }

    async findExistingForUniqueness(
        email: string,
        name: string,
        excludeId?: string,
        transaction?: RepositoryTransaction,
    ): Promise<UserModel[]> {
        const dbClient = getTransactionClient(transaction, this.db);
        const lowerEmail = email.toLowerCase();
        const condition = excludeId
            ? and(or(eq(users.email, lowerEmail), eq(users.name, name)), ne(users.id, excludeId))
            : or(eq(users.email, lowerEmail), eq(users.name, name));

        const foundUsers = await dbClient.select().from(users).where(condition);

        return foundUsers.map(user => this.mapToUser(user));
    }

    private mapToUser(user: UserSelect): UserModel {
        return {
            ...user,
            tokens: [],
        };
    }
}
