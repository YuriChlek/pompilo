import { mailSettings } from '@/module-mail/schemas/mail-settings.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('MailSettingsSchema', () => {
    it('should have correct table name', () => {
        const config = getTableConfig(mailSettings);
        expect(config.name).toBe('mail_settings');
    });

    it('should have a singleton_key with unique constraint', () => {
        const columns = getTableConfig(mailSettings).columns;
        const singletonKey = columns.find(c => c.name === 'singleton_key');

        expect(singletonKey).toBeDefined();
        expect(singletonKey?.isUnique).toBe(true);
        expect(singletonKey?.notNull).toBe(true);
        expect(singletonKey?.default).toBe(true);
    });

    it('should have correct columns and types', () => {
        const columns = getTableConfig(mailSettings).columns;

        const columnNames = columns.map(c => c.name);
        expect(columnNames).toContain('mail_settings_id');
        expect(columnNames).toContain('provider');
        expect(columnNames).toContain('smtp_host');
        expect(columnNames).toContain('smtp_port');
        expect(columnNames).toContain('smtp_secure');
        expect(columnNames).toContain('smtp_user');
        expect(columnNames).toContain('smtp_password_encrypted');
        expect(columnNames).toContain('from_address');
        expect(columnNames).toContain('from_name');
        expect(columnNames).toContain('enabled');
    });

    it('should have foreign key references to users table', () => {
        const foreignKeys = getTableConfig(mailSettings).foreignKeys;

        const confirmedByFk = foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'confirmed_by_user_id'),
        );

        expect(confirmedByFk).toBeDefined();
        expect(getTableConfig(confirmedByFk!.reference().foreignTable).name).toBe('users');
    });
});
