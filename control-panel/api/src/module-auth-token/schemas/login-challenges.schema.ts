import {
    check,
    index,
    integer,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';

export const loginChallenges = pgTable(
    'login_challenges',
    {
        id: uuid('login_challenge_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        realm: varchar('realm', { length: 255 }).notNull(),
        knownDeviceId: uuid('known_device_id').references(() => knownDevices.id, {
            onDelete: 'cascade',
        }),
        deviceId: uuid('device_id').notNull(),
        challengeType: varchar('challenge_type', { length: 255 }).notNull().default('email_code'),
        checkpointTokenHash: text('checkpoint_token_hash').notNull(),
        codeHash: text('code_hash').notNull(),
        attemptCount: integer('attempt_count').notNull().default(0),
        maxAttempts: integer('max_attempts').notNull().default(5),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        approvedAt: timestamp('approved_at', { withTimezone: true }),
        consumedAt: timestamp('consumed_at', { withTimezone: true }),
        failedAt: timestamp('failed_at', { withTimezone: true }),
        expiredAt: timestamp('expired_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        ipAddress: varchar('ip_address', { length: 45 }),
        country: varchar('country', { length: 2 }),
        region: varchar('region', { length: 128 }),
        city: varchar('city', { length: 128 }),
        userAgent: text('user_agent'),
        riskScore: integer('risk_score').notNull().default(0),
        riskReason: text('risk_reason'),
    },
    table => ({
        loginChallengesRealmCheck: check(
            'login_challenges_realm_check',
            sql`${table.realm} in ('customer', 'admin')`,
        ),
        loginChallengesAttemptsCheck: check(
            'login_challenges_attempts_check',
            sql`${table.attemptCount} >= 0 and ${table.maxAttempts} > 0 and ${table.attemptCount} <= ${table.maxAttempts}`,
        ),
        loginChallengesActiveUnique: uniqueIndex('login_challenges_active_unique')
            .on(table.userId, table.realm, table.deviceId)
            .where(sql`consumed_at is null and failed_at is null and expired_at is null`),
        loginChallengesExpiryIdx: index('login_challenges_expiry_idx').on(table.expiresAt),
    }),
);

export type LoginChallengeSelect = typeof loginChallenges.$inferSelect;
export type LoginChallengeInsert = typeof loginChallenges.$inferInsert;

export const isLoginChallengeActive = (
    challenge: LoginChallengeSelect,
    now = new Date(),
): boolean => {
    return (
        challenge.consumedAt === null &&
        challenge.failedAt === null &&
        challenge.expiredAt === null &&
        challenge.expiresAt.getTime() > now.getTime() &&
        challenge.attemptCount < challenge.maxAttempts
    );
};

export const isLoginChallengeFailed = (challenge: LoginChallengeSelect): boolean => {
    return challenge.failedAt !== null || challenge.attemptCount >= challenge.maxAttempts;
};

export const isLoginChallengeExpired = (
    challenge: LoginChallengeSelect,
    now = new Date(),
): boolean => {
    return challenge.expiredAt !== null || challenge.expiresAt.getTime() <= now.getTime();
};

export const isLoginChallengeConsumed = (challenge: LoginChallengeSelect): boolean => {
    return challenge.consumedAt !== null;
};

export const isLoginChallengeApproved = (challenge: LoginChallengeSelect): boolean => {
    return challenge.approvedAt !== null;
};
