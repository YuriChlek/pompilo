import { reauthConfirmations } from '@/module-auth-token/schemas/reauth-confirmations.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('ReauthConfirmationsSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(reauthConfirmations);
        expect(config.name).toBe('reauth_confirmations');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(reauthConfirmations);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('reauth_confirmation_id');
        expect(columnNames).toContain('user_id');
        expect(columnNames).toContain('realm');
        expect(columnNames).toContain('session_id');
        expect(columnNames).toContain('action_scope');
        expect(columnNames).toContain('confirmation_token_hash');
        expect(columnNames).toContain('expires_at');
        expect(columnNames).toContain('consumed_at');
        expect(columnNames).toContain('created_at');
    });

    it('should have correct foreign key referencing users', () => {
        const config = getTableConfig(reauthConfirmations);
        expect(config.foreignKeys.length).toBeGreaterThan(0);

        const userFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'user_id'),
        );
        expect(userFk).toBeDefined();
        expect(userFk?.onDelete).toBe('cascade');
    });

    it('should have correct foreign key referencing sessions', () => {
        const config = getTableConfig(reauthConfirmations);
        const sessionFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'session_id'),
        );
        expect(sessionFk).toBeDefined();
        expect(sessionFk?.onDelete).toBe('cascade');
    });

    it('should have correct unique and btree indexes', () => {
        const config = getTableConfig(reauthConfirmations);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('reauth_confirmations_token_hash_unique');
        expect(indexNames).toContain('reauth_confirmations_expiry_idx');

        const uniqueIdx = indexes.find(
            idx => idx.config.name === 'reauth_confirmations_token_hash_unique',
        );
        expect(uniqueIdx).toBeDefined();
        expect(uniqueIdx?.config.unique).toBe(true);

        const expiryIdx = indexes.find(
            idx => idx.config.name === 'reauth_confirmations_expiry_idx',
        );
        expect(expiryIdx).toBeDefined();
        expect(expiryIdx?.config.unique).toBeFalsy();
    });

    it('should restrict realm values at the database level', () => {
        const config = getTableConfig(reauthConfirmations);
        expect(config.checks.map(check => check.name)).toContain(
            'reauth_confirmations_realm_check',
        );
    });
});
