/**
 * Creates an admin user for the centralized CLI.
 *
 * Run from api/:
 *   npm run cli -- admin:create --admin-email=admin@example.com --admin-password=adminpass123 --admin-firstname=Admin --admin-lastname=Name
 *
 * Optional env vars: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import pg from 'pg';
import { Argon2HashUtil } from '../../../src/common/utils/hash.util';

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

export async function createAdminUser(
    args: Record<string, string>,
): Promise<{ userId: string; name: string; email: string; role: string }> {
    loadEnvFile();

    const dbConfig = getDbConfig();
    const missingEnv = Object.entries(dbConfig)
        .filter(([, value]) => value === undefined || value === '' || Number.isNaN(value))
        .map(([key]) => key);

    if (missingEnv.length > 0) {
        throw new Error(`Missing database configuration: ${missingEnv.join(', ')}`);
    }

    // Support Magento-style parameters and standard options
    const email = args.email || args['admin-email'];
    const password = args.password || args['admin-password'];

    let name = args.name || args['admin-user'] || args.username;
    if (
        !name &&
        (args.firstname || args['admin-firstname'] || args.lastname || args['admin-lastname'])
    ) {
        const first = args.firstname || args['admin-firstname'] || '';
        const last = args.lastname || args['admin-lastname'] || '';
        name = `${first} ${last}`.trim();
    }

    const role = args.role || 'admin';

    if (!email || !password || !name) {
        throw new Error(
            'Missing required arguments: email, password, and name/firstname/lastname must be provided.',
        );
    }

    if (role !== 'admin' && role !== 'superAdmin') {
        throw new Error('Invalid role. Role must be either "admin" or "superAdmin".');
    }

    const client = new pg.Client(dbConfig);
    await client.connect();

    try {
        // Check if user already exists
        const checkRes = await client.query<{ user_id: string }>(
            'select user_id from "users" where "email" = $1',
            [email],
        );
        if (checkRes.rows.length > 0) {
            throw new Error(`User with email "${email}" already exists.`);
        }

        const hashedPassword = await Argon2HashUtil.hash(password);

        await client.query('begin');
        const insertQuery = `
            insert into "users" ("name", "email", "password", "role", "is_active", "email_verified_at")
            values ($1, $2, $3, $4, $5, now())
            returning "user_id"
        `;
        const insertRes = await client.query<{ user_id: string }>(insertQuery, [
            name,
            email,
            hashedPassword,
            role,
            true,
        ]);
        const userId = insertRes.rows[0]?.user_id;
        await client.query('commit');

        return {
            userId,
            name,
            email,
            role,
        };
    } catch (error) {
        await client.query('rollback');
        throw error;
    } finally {
        await client.end();
    }
}
