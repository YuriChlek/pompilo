import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

describe('mail outbox production migration settings', () => {
    const migrationDirectory = join(process.cwd(), 'drizzle-migrations');
    const migrationMetaDirectory = join(process.cwd(), 'drizzle-migrations/meta');
    const baselineMigrationFile = readdirSync(migrationDirectory).find(file =>
        /^0000_.*\.sql$/.test(file),
    );
    if (!baselineMigrationFile) {
        throw new Error('Baseline migration 0000_*.sql was not found.');
    }

    const baselineMigration = readFileSync(
        join(migrationDirectory, baselineMigrationFile),
        'utf8',
    );

    it('should keep partial indexes aligned with claim and stale reclaim queries', () => {
        expect(baselineMigration).toContain(
            'CREATE INDEX "mail_outbox_claim_idx" ON "mail_outbox" USING btree ("priority" DESC NULLS LAST,"available_at") WHERE status = \'pending\'',
        );
        expect(baselineMigration).toContain(
            'CREATE INDEX "mail_outbox_stale_reclaim_idx" ON "mail_outbox" USING btree ("locked_at") WHERE status = \'pending\' AND locked_at IS NOT NULL',
        );
    });

    it('should configure aggressive autovacuum for the hot outbox table', () => {
        expect(baselineMigration).toContain('ALTER TABLE "mail_outbox" SET');
        expect(baselineMigration).toContain('autovacuum_vacuum_scale_factor = 0.05');
        expect(baselineMigration).toContain('autovacuum_vacuum_threshold = 100');
    });

    it('should enforce exactly one payload storage column in PostgreSQL', () => {
        expect(baselineMigration).toContain('mail_outbox_payload_exactly_one_check');
        expect(baselineMigration).toContain(
            '(payload_encrypted IS NOT NULL) <> (payload_json IS NOT NULL)',
        );
    });

    it('should keep a complete snapshot chain through the latest migration', () => {
        const snapshot = JSON.parse(
            readFileSync(join(migrationMetaDirectory, '0000_snapshot.json'), 'utf8'),
        ) as {
            id: string;
            prevId: string;
            tables: Record<
                string,
                {
                    checkConstraints?: Record<string, unknown>;
                }
            >;
        };

        expect(snapshot.prevId).toBe('00000000-0000-0000-0000-000000000000');
        expect(snapshot.tables['public.mail_outbox'].checkConstraints).toHaveProperty(
            'mail_outbox_payload_exactly_one_check',
        );
    });
});
