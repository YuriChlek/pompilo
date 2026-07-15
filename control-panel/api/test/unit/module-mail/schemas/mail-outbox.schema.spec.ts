import { mailOutbox, mailOutboxStatusEnum } from '@/module-mail/schemas/mail-outbox.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('MailOutboxSchema', () => {
    it('should have correct table name', () => {
        const config = getTableConfig(mailOutbox);
        expect(config.name).toBe('mail_outbox');
    });

    it('should have correct columns and types', () => {
        const columns = getTableConfig(mailOutbox).columns;
        const columnNames = columns.map(c => c.name);

        expect(columnNames).toContain('mail_outbox_id');
        expect(columnNames).toContain('idempotency_key');
        expect(columnNames).toContain('status');
        expect(columnNames).toContain('payload_encrypted');
        expect(columnNames).toContain('payload_json');
        expect(columnNames).toContain('priority');
        expect(columnNames).toContain('attempt_count');
        expect(columnNames).toContain('last_error');
        expect(columnNames).toContain('available_at');
        expect(columnNames).toContain('locked_at');
        expect(columnNames).toContain('locked_by');
        expect(columnNames).toContain('queued_job_id');
        expect(columnNames).toContain('queued_at');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('updated_at');
    });

    it('should have a status enum with correct values', () => {
        expect(mailOutboxStatusEnum.enumValues).toEqual(['pending', 'queued', 'sent', 'failed']);
    });

    it('should have correct indexes defined', () => {
        const config = getTableConfig(mailOutbox);
        const indexes = config.indexes.map(idx => idx.config.name);

        expect(indexes).toContain('mail_outbox_claim_idx');
        expect(indexes).toContain('mail_outbox_stale_reclaim_idx');
    });

    it('should require exactly one payload storage column', () => {
        const checks = getTableConfig(mailOutbox).checks.map(item => item.name);

        expect(checks).toContain('mail_outbox_payload_exactly_one_check');
    });
});
