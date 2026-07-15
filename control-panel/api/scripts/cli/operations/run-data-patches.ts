import {
    existsSync as fsExistsSync,
    readFileSync as fsReadFileSync,
    readdirSync as fsReaddirSync,
} from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, basename, join } from 'node:path';
import * as crypto from 'node:crypto';
import pg from 'pg';
import type {
    DataPatch,
    DataPatchContext,
} from '../../../src/module-data-patch/types/data-patch.types';
import { getErrorMessage } from '../error-format';

const dynamicRequire = createRequire(__filename);
const OPERATIONS_ROOT = resolve(__dirname, '../../..');

// Env loading
type EnvFileReader = (path: string) => string;
type ChecksumFileReader = (path: string) => Buffer;

function loadEnvFile(
    existsSyncFn = fsExistsSync,
    readFileSyncFn: EnvFileReader = path => fsReadFileSync(path, 'utf8'),
): void {
    const candidates = [
        resolve(process.cwd(), '.env'),
        resolve(process.cwd(), '.env.development'),
        resolve(process.cwd(), 'api/.env'),
        resolve(process.cwd(), 'api/.env.development'),
        resolve(process.cwd(), '../.env'),
        resolve(process.cwd(), '../.env.development'),
    ];
    const uniqueExistingPaths = Array.from(new Set(candidates.filter(path => existsSyncFn(path))));

    for (const envPath of uniqueExistingPaths) {
        const content = readFileSyncFn(envPath);
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

function calculateChecksum(
    filePath: string,
    readFileSyncFn: ChecksumFileReader = path => fsReadFileSync(path),
): string {
    const fileBuffer = readFileSyncFn(filePath);
    const hashSum = crypto.createHash('sha256');
    hashSum.update(fileBuffer);
    return `sha256:${hashSum.digest('hex')}`;
}

export type FsOverrides = {
    existsSync?: (path: string) => boolean;
    readFileSync?: (path: string, options?: BufferEncoding | null) => string | Buffer;
    readdirSync?: (path: string) => string[];
    requireOverride?: <T = unknown>(path: string) => T;
    mode?: 'source' | 'compiled';
};

type DataPatchManifestEntry = {
    name: string;
    sourcePath: string;
    compiledPath: string;
    sourceChecksum: string;
};

type TableExistsRow = {
    exists: boolean;
};

type AppliedPatchRow = {
    patchName: string;
    checksum: string;
};

type DataPatchModule = {
    patch?: DataPatch;
};

function requireModule<T = unknown>(path: string): T {
    return dynamicRequire(path) as T;
}

function isManifestEntry(value: unknown): value is DataPatchManifestEntry {
    if (!value || typeof value !== 'object') {
        return false;
    }

    const entry = value as Record<string, unknown>;

    return (
        typeof entry.name === 'string' &&
        typeof entry.sourcePath === 'string' &&
        typeof entry.compiledPath === 'string' &&
        typeof entry.sourceChecksum === 'string'
    );
}

export async function runDataPatches(
    argv: string[],
    poolOverride?: pg.Pool,
    fsOverrides?: FsOverrides,
): Promise<{ total: number; skipped: number; applied: number }> {
    const existsSyncFn = fsOverrides?.existsSync || fsExistsSync;
    const readFileSyncFn = fsOverrides?.readFileSync || fsReadFileSync;
    const readdirSyncFn = fsOverrides?.readdirSync || fsReaddirSync;
    const requireFn = fsOverrides?.requireOverride ?? requireModule;

    loadEnvFile(existsSyncFn, (path: string) => readFileSyncFn(path, 'utf8').toString());

    const isDryRun = argv.includes('--dry-run');
    const isList = argv.includes('--list');

    // Validate DB credentials
    const host = process.env.DB_HOST;
    const port = process.env.DB_PORT;
    const user = process.env.DB_USER;
    const password = process.env.DB_PASSWORD;
    const database = process.env.DB_NAME;

    if (!host || !user || !database) {
        throw new Error(
            `Database configuration environment variables are missing. DB_HOST: ${host}, DB_USER: ${user}, DB_NAME: ${database}`,
        );
    }

    const pool =
        poolOverride ||
        new pg.Pool({
            host,
            port: port ? parseInt(port, 10) : 5432,
            user,
            password,
            database,
        });

    let lockClient: pg.PoolClient | null = null;
    try {
        // Connect and acquire advisory lock
        lockClient = await pool.connect();
        await lockClient.query("SELECT pg_advisory_lock(hashtext('pampilo_data_patches_runner'))");

        // Verify table exists
        const tableCheck = await lockClient.query<TableExistsRow>(`
            SELECT EXISTS (
                SELECT FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename  = 'data_patches'
            );
        `);
        const tableExists = tableCheck.rows[0]?.exists;
        if (!tableExists) {
            throw new Error(
                'Table "data_patches" does not exist in the database. Run migrations first.',
            );
        }

        // Discover patches
        interface DiscoveredPatch {
            name: string;
            sourceChecksum: string;
            filePath: string;
        }

        const discoveredPatches: DiscoveredPatch[] = [];
        const isSourceMode = fsOverrides?.mode
            ? fsOverrides.mode === 'source'
            : __filename.endsWith('.ts');

        if (isSourceMode) {
            const dataPatchesDir = resolve(OPERATIONS_ROOT, 'data-patches');
            if (!existsSyncFn(dataPatchesDir)) {
                throw new Error(`Data patches directory not found at ${dataPatchesDir}`);
            }
            const files = readdirSyncFn(dataPatchesDir);
            for (const file of files) {
                if (!file.endsWith('.ts') || file.endsWith('.spec.ts')) {
                    continue;
                }
                const baseName = basename(file, '.ts');
                const filePath = join(dataPatchesDir, file);
                const sourceChecksum = calculateChecksum(filePath, (path: string) => {
                    const content = readFileSyncFn(path, null);
                    return typeof content === 'string' ? Buffer.from(content) : content;
                });
                discoveredPatches.push({
                    name: baseName,
                    sourceChecksum,
                    filePath,
                });
            }
        } else {
            // Compiled mode
            const manifestPath = resolve(OPERATIONS_ROOT, 'data-patches-manifest.json');
            if (!existsSyncFn(manifestPath)) {
                throw new Error(`Data patches manifest not found at ${manifestPath}`);
            }
            const manifestContent = JSON.parse(readFileSyncFn(manifestPath, 'utf8') as string) as {
                patches?: unknown;
            };
            if (!Array.isArray(manifestContent.patches)) {
                throw new Error(
                    `Data patches manifest at ${manifestPath} must contain a patches array.`,
                );
            }

            for (const entry of manifestContent.patches) {
                if (!isManifestEntry(entry)) {
                    throw new Error(`Invalid data patch manifest entry in ${manifestPath}.`);
                }
                const filePath = resolve(OPERATIONS_ROOT, entry.compiledPath);
                if (!existsSyncFn(filePath)) {
                    throw new Error(`Compiled data patch file not found at ${filePath}`);
                }
                discoveredPatches.push({
                    name: entry.name,
                    sourceChecksum: entry.sourceChecksum,
                    filePath,
                });
            }
        }

        // Sort by filename/name ascending
        discoveredPatches.sort((a, b) => a.name.localeCompare(b.name));

        // Validate duplicates and names
        const patchNameRegex = /^\d{12}-[a-z0-9-]+$/;
        const nameSet = new Set<string>();
        for (const dp of discoveredPatches) {
            if (nameSet.has(dp.name)) {
                throw new Error(`Duplicate patch name found: "${dp.name}"`);
            }
            nameSet.add(dp.name);

            if (!patchNameRegex.test(dp.name)) {
                throw new Error(
                    `Patch name "${dp.name}" does not match naming convention YYYYMMDDHHMM-short-kebab-description`,
                );
            }
        }

        // Load applied patches
        const appliedPatchesRes = await lockClient.query<AppliedPatchRow>(
            'SELECT patch_name as "patchName", checksum FROM data_patches',
        );
        const appliedPatchesMap = new Map<string, string>();
        for (const row of appliedPatchesRes.rows) {
            appliedPatchesMap.set(row.patchName, row.checksum);
        }

        if (isList) {
            console.log('--- List of Data Patches ---');
            for (const dp of discoveredPatches) {
                const appliedChecksum = appliedPatchesMap.get(dp.name);
                if (appliedChecksum) {
                    const status =
                        appliedChecksum === dp.sourceChecksum
                            ? 'Applied'
                            : 'Drifted (Checksum Mismatch)';
                    console.log(`[${status}] ${dp.name}`);
                } else {
                    console.log(`[Pending] ${dp.name}`);
                }
            }
            return { total: discoveredPatches.length, skipped: 0, applied: 0 };
        }

        // Validate checksum drifts for already applied patches
        for (const dp of discoveredPatches) {
            const appliedChecksum = appliedPatchesMap.get(dp.name);
            if (appliedChecksum && appliedChecksum !== dp.sourceChecksum) {
                throw new Error(
                    `Checksum drift detected for already applied patch "${dp.name}". Database: ${appliedChecksum}, Local source: ${dp.sourceChecksum}. Applied patches cannot be modified. Create a new patch instead.`,
                );
            }
        }

        if (isDryRun) {
            console.log('--- Dry Run Mode ---');
            let pendingCount = 0;
            for (const dp of discoveredPatches) {
                if (!appliedPatchesMap.has(dp.name)) {
                    console.log(`[Dry Run] Pending patch: ${dp.name}`);
                    pendingCount++;
                }
            }
            console.log(
                `Summary: ${discoveredPatches.length} patches found, ${pendingCount} pending.`,
            );
            return {
                total: discoveredPatches.length,
                skipped: discoveredPatches.length - pendingCount,
                applied: 0,
            };
        }

        // Normal Run Mode
        console.log(`Found ${discoveredPatches.length} patches total.`);
        let skippedCount = 0;
        let appliedCount = 0;

        for (const dp of discoveredPatches) {
            if (appliedPatchesMap.has(dp.name)) {
                skippedCount++;
                continue;
            }

            // Load and execute patch
            let patchModule: DataPatchModule;
            try {
                patchModule = requireFn<DataPatchModule>(dp.filePath);
            } catch (err) {
                throw new Error(
                    `Failed to import patch file ${dp.filePath}: ${getErrorMessage(err)}`,
                );
            }

            const patch = patchModule.patch;
            if (!patch) {
                throw new Error(`Patch file "${dp.filePath}" does not export "patch" object.`);
            }
            if (patch.name !== dp.name) {
                throw new Error(
                    `Exported patch name "${patch.name}" does not match file name "${dp.name}"`,
                );
            }

            const patchClient = await pool.connect();
            const startTime = Date.now();
            try {
                await patchClient.query('BEGIN');

                const context: DataPatchContext = {
                    client: patchClient,
                    logger: {
                        info: (msg: string) => console.log(`[${patch.name}] INFO: ${msg}`),
                        warn: (msg: string) => console.warn(`[${patch.name}] WARN: ${msg}`),
                        error: (msg: string) => console.error(`[${patch.name}] ERROR: ${msg}`),
                    },
                    env: process.env,
                };

                console.log(`Applying data patch: ${patch.name}...`);
                await patch.apply(context);

                const durationMs = Date.now() - startTime;

                await patchClient.query(
                    `INSERT INTO data_patches (patch_name, checksum, duration_ms, description, app_version, node_env)
                     VALUES ($1, $2, $3, $4, $5, $6)`,
                    [
                        patch.name,
                        dp.sourceChecksum,
                        durationMs,
                        patch.description || null,
                        process.env.APP_VERSION || null,
                        process.env.NODE_ENV || 'development',
                    ],
                );

                await patchClient.query('COMMIT');
                console.log(`Successfully applied patch: ${patch.name} (${durationMs}ms)`);
                appliedCount++;
            } catch (err) {
                await patchClient.query('ROLLBACK');
                throw new Error(
                    `Failed to apply data patch "${patch.name}". Transaction rolled back. Error: ${getErrorMessage(err)}`,
                );
            } finally {
                patchClient.release();
            }
        }

        console.log('--- Run Summary ---');
        console.log(`Total patches: ${discoveredPatches.length}`);
        console.log(`Already applied (skipped): ${skippedCount}`);
        console.log(`Newly applied: ${appliedCount}`);

        return { total: discoveredPatches.length, skipped: skippedCount, applied: appliedCount };
    } finally {
        if (lockClient) {
            try {
                await lockClient.query(
                    "SELECT pg_advisory_unlock(hashtext('pampilo_data_patches_runner'))",
                );
            } catch (unlockErr) {
                console.error('Failed to release advisory lock:', unlockErr);
            }
            lockClient.release();
        }
        if (!poolOverride) {
            await pool.end();
        }
    }
}
