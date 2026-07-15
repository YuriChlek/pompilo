import { Inject, Injectable } from '@nestjs/common';
import { eq } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    userSettings,
    UserSettingsInsert,
    UserSettingsSelect,
} from '@/module-account/schemas/user-settings.schema';

@Injectable()
export class UserSettingsRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async findByUserId(
        userId: string,
        tx?: NodePgDatabase<typeof schema>,
    ): Promise<UserSettingsSelect | null> {
        const db = tx ?? this.db;
        const [settings] = await db
            .select()
            .from(userSettings)
            .where(eq(userSettings.userId, userId))
            .limit(1);

        return settings ?? null;
    }

    async create(
        data: UserSettingsInsert,
        tx?: NodePgDatabase<typeof schema>,
    ): Promise<UserSettingsSelect> {
        const db = tx ?? this.db;
        const [settings] = await db.insert(userSettings).values(data).returning();

        return settings;
    }

    async update(
        userId: string,
        data: Partial<UserSettingsInsert>,
        tx?: NodePgDatabase<typeof schema>,
    ): Promise<UserSettingsSelect> {
        const db = tx ?? this.db;
        const [settings] = await db
            .update(userSettings)
            .set({
                ...data,
                updatedAt: new Date(),
            })
            .where(eq(userSettings.userId, userId))
            .returning();

        return settings;
    }

    async createOnConflictDoNothing(
        userId: string,
        tx?: NodePgDatabase<typeof schema>,
    ): Promise<void> {
        const db = tx ?? this.db;
        await db.insert(userSettings).values({ userId }).onConflictDoNothing();
    }
}
