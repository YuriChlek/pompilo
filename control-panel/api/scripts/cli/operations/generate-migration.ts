/**
 * Generates a database migration schema file for the centralized CLI.
 *
 * Run from api/: npm run cli -- db:migration:generate [--name=<migration-name>]
 * Optional env vars: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const REQUIRED_BASELINE_SQL = `
--> statement-breakpoint
CREATE OR REPLACE FUNCTION notify_mail_outbox_inserted()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('mail_outbox_inserted', NEW.mail_outbox_id::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
--> statement-breakpoint
DROP TRIGGER IF EXISTS mail_outbox_inserted_trigger ON mail_outbox;
--> statement-breakpoint
CREATE TRIGGER mail_outbox_inserted_trigger
AFTER INSERT ON mail_outbox
FOR EACH ROW
EXECUTE FUNCTION notify_mail_outbox_inserted();
--> statement-breakpoint
ALTER TABLE "mail_outbox" SET (
\tautovacuum_vacuum_scale_factor = 0.05,
\tautovacuum_vacuum_threshold = 100
);
`;

function runDrizzleGenerate(args: string[]): void {
    const result = spawnSync('npx', ['drizzle-kit', 'generate', ...args], {
        cwd: getProjectRoot(),
        stdio: 'inherit',
    });

    if (result.status !== 0) {
        throw new Error(`Drizzle kit generate failed with status: ${result.status}`);
    }
}

function getOutputDirectory(args: string[]): string {
    const outFlagIndex = args.findIndex(arg => arg === '--out');
    const configuredOut =
        outFlagIndex >= 0 && args[outFlagIndex + 1] ? args[outFlagIndex + 1] : 'drizzle-migrations';

    return path.resolve(getProjectRoot(), configuredOut);
}

function getProjectRoot(): string {
    return __dirname.includes('dist')
        ? path.resolve(__dirname, '../../../..')
        : path.resolve(__dirname, '../../..');
}

function getSqlFiles(directory: string): Set<string> {
    if (!existsSync(directory)) {
        return new Set();
    }

    return new Set(readdirSync(directory).filter(fileName => fileName.endsWith('.sql')));
}

function appendRequiredBaselineSql(
    outputDirectory: string,
    previousSqlFiles: Set<string>,
    hadMetadata: boolean,
    args: string[],
): void {
    if (hadMetadata || args.includes('--custom')) {
        return;
    }

    const generatedSqlFiles = [...getSqlFiles(outputDirectory)].filter(
        fileName => !previousSqlFiles.has(fileName),
    );
    if (generatedSqlFiles.length !== 1) {
        throw new Error(
            `Expected one fresh baseline migration, received ${generatedSqlFiles.length}.`,
        );
    }

    appendFileSync(path.join(outputDirectory, generatedSqlFiles[0]), REQUIRED_BASELINE_SQL);
}

export function generateMigration(args: string[]): void {
    const outputDirectory = getOutputDirectory(args);
    const previousSqlFiles = getSqlFiles(outputDirectory);
    const hadMetadata = existsSync(path.join(outputDirectory, 'meta', '_journal.json'));

    runDrizzleGenerate(args);
    appendRequiredBaselineSql(outputDirectory, previousSqlFiles, hadMetadata, args);
}
