/**
 * Creates demo identity users directly in Postgres for the centralized CLI.
 *
 * Optional env:
 *   DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
 *   SEED_USER_PASSWORD
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import pg from 'pg';
import { Argon2HashUtil } from '../../../../src/common/utils/hash.util';
import { UserRoles } from '../../../../src/module-auth/enums/auth.enums';

type DbConfig = {
    host: string | undefined;
    port: number;
    user: string | undefined;
    password: string | undefined;
    database: string | undefined;
};

type SeedUser = {
    name: string;
    email: string;
    password: string;
    role: UserRoles.USER;
    tenantName: string;
};

type ExistingUserRow = {
    user_id: string;
    name: string;
    email: string;
    role: UserRoles;
};

type CreatedUserResult =
    | { status: 'created'; userId: string }
    | { status: 'skipped'; reason: string };

export type SeedUsersSummary = {
    created: number;
    skipped: number;
    failed: number;
};

const DEMO_USER_NUMBER = 10;

function buildSeedUsers(password: string): SeedUser[] {
    return Array.from({ length: DEMO_USER_NUMBER }, (_, index) => {
        const number = index + 1;
        return {
            name: `Demo User ${number}`,
            email: `user${number}@seed.dev`,
            password,
            role: UserRoles.USER,
            tenantName: `Demo User ${number}'s Space`,
        };
    });
}

export function loadEnvFile(): void {
    const candidates = [
        resolve(process.cwd(), '.env'),
        resolve(process.cwd(), '.env.development'),
        resolve(process.cwd(), 'api/.env'),
        resolve(process.cwd(), 'api/.env.development'),
        resolve(process.cwd(), '../.env'),
        resolve(process.cwd(), '../.env.development'),
    ];
    const envPaths = Array.from(new Set(candidates.filter(path => existsSync(path))));

    for (const envPath of envPaths) {
        const content = readFileSync(envPath, 'utf8');

        for (const line of content.split(/\r?\n/)) {
            const trimmed = line.trim();

            if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) {
                continue;
            }

            const [key, ...valueParts] = trimmed.split('=');
            if (key) {
                process.env[key.trim()] ??= valueParts
                    .join('=')
                    .trim()
                    .replace(/^['"]|['"]$/g, '');
            }
        }
    }
}

export function getDbConfig(): DbConfig {
    return {
        host: process.env.DB_HOST ?? 'localhost',
        port: Number(process.env.DB_PORT ?? 5432),
        user: process.env.DB_USER ?? 'admin',
        password: process.env.DB_PASSWORD ?? 'admin_pass',
        database: process.env.DB_NAME ?? 'pampilo_db',
    };
}

export function assertDbConfig(dbConfig: DbConfig): asserts dbConfig is Required<DbConfig> {
    const missing = Object.entries(dbConfig)
        .filter(([, value]) => value === undefined || value === '' || Number.isNaN(value))
        .map(([key]) => key);

    if (missing.length > 0) {
        throw new Error(`Missing database configuration: ${missing.join(', ')}`);
    }
}

async function findExistingUser(
    client: pg.Client,
    user: SeedUser,
): Promise<ExistingUserRow | null> {
    const result = await client.query<ExistingUserRow>(
        `
            select "user_id", "name", "email", "role"
            from "users"
            where "email" = $1 or "name" = $2
            limit 1
        `,
        [user.email, user.name],
    );

    return result.rows[0] ?? null;
}

async function insertBaseUser(
    client: pg.Client,
    user: SeedUser,
    passwordHash: string,
): Promise<string> {
    const result = await client.query<{ user_id: string }>(
        `
            insert into "users" ("name", "email", "password", "role", "is_active")
            values ($1, $2, $3, $4, true)
            returning "user_id"
        `,
        [user.name, user.email, passwordHash, user.role],
    );

    const userId = result.rows[0]?.user_id;
    if (!userId) {
        throw new Error(`User insert did not return an id for ${user.email}`);
    }

    return userId;
}

async function ensureTenantMembership(
    client: pg.Client,
    userId: string,
    user: SeedUser,
): Promise<void> {
    const tenantResult = await client.query<{ tenant_id: string }>(
        `
            insert into "tenants" ("name", "status")
            values ($1, 'ACTIVE')
            returning "tenant_id"
        `,
        [user.tenantName],
    );

    const tenantId = tenantResult.rows[0]?.tenant_id;
    if (!tenantId) {
        throw new Error(`Tenant insert did not return an id for ${user.email}`);
    }

    await client.query(
        `
            insert into "memberships" ("user_id", "tenant_id", "role")
            values ($1, $2, 'OWNER')
            on conflict ("user_id", "tenant_id") do nothing
        `,
        [userId, tenantId],
    );
}

async function createUser(client: pg.Client, user: SeedUser): Promise<CreatedUserResult> {
    const existing = await findExistingUser(client, user);
    if (existing) {
        return {
            status: 'skipped',
            reason: `already exists as ${existing.email} (${existing.role})`,
        };
    }

    await client.query('begin');
    try {
        const passwordHash = await Argon2HashUtil.hash(user.password);
        const userId = await insertBaseUser(client, user, passwordHash);

        await ensureTenantMembership(client, userId, user);
        await client.query('commit');

        return { status: 'created', userId };
    } catch (error) {
        await client.query('rollback');
        throw error;
    }
}

export async function seedDemoUsers(): Promise<SeedUsersSummary> {
    loadEnvFile();

    const dbConfig = getDbConfig();
    assertDbConfig(dbConfig);

    const users = buildSeedUsers(process.env.SEED_USER_PASSWORD ?? 'Seed1234');
    const client = new pg.Client(dbConfig);
    await client.connect();

    console.log(
        `Seeding users directly into ${dbConfig.database} at ${dbConfig.host}:${dbConfig.port}\n`,
    );

    let created = 0;
    let skipped = 0;
    let failed = 0;

    try {
        for (const user of users) {
            try {
                const result = await createUser(client, user);

                if (result.status === 'created') {
                    console.log(`  +  ${user.role.padEnd(7)}  ${user.email}`);
                    created++;
                } else {
                    console.log(`  =  ${user.role.padEnd(7)}  ${user.email}  (${result.reason})`);
                    skipped++;
                }
            } catch (error) {
                console.log(
                    `  x  ${user.role.padEnd(7)}  ${user.email}  -  ${
                        error instanceof Error ? error.message : String(error)
                    }`,
                );
                failed++;
            }
        }
    } finally {
        await client.end();
    }

    console.log(`\nUsers done: ${created} created, ${skipped} skipped, ${failed} failed.`);

    return { created, skipped, failed };
}
