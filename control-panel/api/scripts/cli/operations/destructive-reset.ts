/**
 * Performs a destructive database reset, runs migrations, and seeds default data.
 *
 * Run from api/: npm run cli -- db:destructive-reset
 * Optional env vars: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import pg from 'pg';
import { drizzle } from 'drizzle-orm/node-postgres';
import { migrate } from 'drizzle-orm/node-postgres/migrator';
import { Argon2HashUtil } from '../../../src/common/utils/hash.util';
import { runDataPatches } from './run-data-patches';

const RESET_LOCK_NAME = 'identity_service_destructive_reset';
const PROJECT_ROOT = __dirname.includes('dist')
    ? resolve(__dirname, '../../../..')
    : resolve(__dirname, '../../..');
const MIGRATIONS_DIRECTORY = resolve(PROJECT_ROOT, 'drizzle-migrations');
const PROTECTED_DATABASES = new Set(['postgres', 'template0', 'template1']);
const REQUIRED_PUBLIC_TABLES = [
    'data_patches',
    'email_change_challenges',
    'email_verifications',
    'identity_outbox_events',
    'known_devices',
    'login_challenges',
    'mail_audit_events',
    'mail_outbox',
    'mail_settings',
    'memberships',
    'password_reset_challenges',
    'reauth_confirmations',
    'security_events',
    'sessions',
    'tenants',
    'tokens',
    'user_settings',
    'users',
] as const;

function loadEnvFile(): void {
    const candidates = [
        resolve(process.cwd(), '.env'),
        resolve(process.cwd(), '.env.development'),
        resolve(process.cwd(), 'api/.env'),
        resolve(process.cwd(), 'api/.env.development'),
        resolve(process.cwd(), '../.env'),
        resolve(process.cwd(), '../.env.development'),
    ];
    const uniqueExistingPaths = Array.from(new Set(candidates.filter(path => existsSync(path))));

    for (const envPath of uniqueExistingPaths) {
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

function getDbConfig() {
    return {
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT ?? 5432),
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_NAME,
    };
}

function requireResetConfiguration(database: string): void {
    if (process.env.NODE_ENV === 'production') {
        throw new Error('Destructive reset is disabled when NODE_ENV=production');
    }

    if (PROTECTED_DATABASES.has(database)) {
        throw new Error(`Refusing to reset protected database "${database}"`);
    }
}

function getAdminSeedConfiguration(): {
    adminEmail: string;
    adminName: string;
    adminPassword: string;
} | null {
    const adminEmail = process.env.DESTRUCTIVE_RESET_ADMIN_EMAIL;
    const adminName = process.env.DESTRUCTIVE_RESET_ADMIN_NAME;
    const adminPassword = process.env.DESTRUCTIVE_RESET_ADMIN_PASSWORD;

    if (!adminEmail && !adminName && !adminPassword) {
        return null;
    }

    if (!adminEmail || !adminName || !adminPassword) {
        throw new Error(
            'DESTRUCTIVE_RESET_ADMIN_EMAIL, DESTRUCTIVE_RESET_ADMIN_NAME and DESTRUCTIVE_RESET_ADMIN_PASSWORD must be provided together',
        );
    }

    if (adminPassword.length < 12) {
        throw new Error('DESTRUCTIVE_RESET_ADMIN_PASSWORD must contain at least 12 characters');
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(adminEmail)) {
        throw new Error('DESTRUCTIVE_RESET_ADMIN_EMAIL must be a valid email address');
    }

    return { adminEmail, adminName, adminPassword };
}

function getExpectedMigrationCount(): number {
    const journal = JSON.parse(
        readFileSync(resolve(MIGRATIONS_DIRECTORY, 'meta/_journal.json'), 'utf8'),
    ) as { entries?: unknown[] };

    if (!Array.isArray(journal.entries) || journal.entries.length === 0) {
        throw new Error('Migration journal must contain at least one entry');
    }

    return journal.entries.length;
}

async function runInTransaction(client: pg.Client, work: () => Promise<void>): Promise<void> {
    await client.query('begin');
    try {
        await work();
        await client.query('commit');
    } catch (error) {
        await client.query('rollback');
        throw error;
    }
}

async function resetSchemas(client: pg.Client): Promise<void> {
    await client.query('DROP SCHEMA IF EXISTS public CASCADE');
    await client.query('DROP SCHEMA IF EXISTS drizzle CASCADE');
    await client.query('CREATE SCHEMA public');
    await client.query('GRANT ALL ON SCHEMA public TO public');
}

async function verifyResetResult(
    client: pg.Client,
    expectedMigrationCount: number,
    adminEmail?: string,
): Promise<void> {
    const verification = await client.query<{
        migration_count: string;
    }>(
        `
            select
                (select count(*)::text from drizzle.__drizzle_migrations) as migration_count
        `,
    );

    const result = verification.rows[0];
    if (!result) {
        throw new Error('Post-reset verification failed: no verification row returned');
    }

    const migrationCount = Number(result.migration_count);

    if (migrationCount !== expectedMigrationCount) {
        throw new Error(
            `Post-reset verification failed: expected ${expectedMigrationCount} migrations, got ${migrationCount}`,
        );
    }

    if (adminEmail) {
        const adminVerification = await client.query<{ admin_count: string }>(
            `
                select count(*)::text as admin_count
                from users
                where email = $1 and role in ('platformAdmin', 'superAdmin')
            `,
            [adminEmail],
        );
        const adminCount = Number(adminVerification.rows[0]?.admin_count);

        if (adminCount !== 1) {
            throw new Error(
                `Post-reset verification failed: expected exactly one seeded admin, got ${adminCount}`,
            );
        }
    }

    const tableVerification = await client.query<{ table_name: string }>(
        `
            select table_name
            from information_schema.tables
            where table_schema = 'public'
              and table_name = any($1::text[])
        `,
        [REQUIRED_PUBLIC_TABLES],
    );
    const existingTables = new Set(tableVerification.rows.map(row => row.table_name));
    const missingTables = REQUIRED_PUBLIC_TABLES.filter(table => !existingTables.has(table));

    if (missingTables.length > 0) {
        throw new Error(
            `Post-reset verification failed: missing required tables ${missingTables.join(', ')}`,
        );
    }

}

export async function runDestructiveReset(): Promise<void> {
    loadEnvFile();

    const dbConfig = getDbConfig();
    const missingEnv = Object.entries(dbConfig)
        .filter(([, value]) => value === undefined || value === '' || Number.isNaN(value))
        .map(([key]) => key);

    if (missingEnv.length > 0) {
        throw new Error(`Missing database configuration: ${missingEnv.join(', ')}`);
    }

    const database = dbConfig.database as string;
    requireResetConfiguration(database);
    const adminSeedConfig = getAdminSeedConfiguration();
    const expectedMigrationCount = getExpectedMigrationCount();
    console.log(
        `Connecting to database "${dbConfig.database}" at ${dbConfig.host}:${dbConfig.port}...`,
    );
    const client = new pg.Client(dbConfig);
    await client.connect();

    try {
        await client.query('select pg_advisory_lock(hashtext($1))', [RESET_LOCK_NAME]);
        console.log('Performing destructive reset (dropping public and drizzle schemas)...');
        await resetSchemas(client);
        console.log('Public and drizzle schemas reset successfully.');

        console.log('Running schema migrations...');
        const db = drizzle(client);
        await migrate(db, {
            migrationsFolder: MIGRATIONS_DIRECTORY,
        });
        console.log('Migrations completed successfully.');

        console.log('Running data patches...');
        await runDataPatches([]);
        console.log('Data patches completed successfully.');

        if (adminSeedConfig) {
            console.log('Seeding default admin user...');
            const hashedPassword = await Argon2HashUtil.hash(adminSeedConfig.adminPassword);

            await runInTransaction(client, async () => {
                await client.query(
                    `
                        insert into "users" ("name", "email", "password", "role", "is_active", "email_verified_at")
                        values ($1, $2, $3, 'platformAdmin', true, now())
                    `,
                    [adminSeedConfig.adminName, adminSeedConfig.adminEmail, hashedPassword],
                );
            });
            console.log(`Seeded default admin user: ${adminSeedConfig.adminEmail}`);
        } else {
            console.log(
                'Skipping default admin user seed: admin environment variables are not set.',
            );
        }

        await verifyResetResult(client, expectedMigrationCount, adminSeedConfig?.adminEmail);

        console.log('Destructive reset and reseed completed successfully!');
    } catch (error) {
        console.error('An error occurred during destructive reset/reseed:', error);
        throw error;
    } finally {
        await client
            .query('select pg_advisory_unlock(hashtext($1))', [RESET_LOCK_NAME])
            .catch(() => undefined);
        await client.end();
    }
}
