import { tokens } from '@/module-auth-token/schemas/tokens.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('TokensSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(tokens);
        expect(config.name).toBe('tokens');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(tokens);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('token_id');
        expect(columnNames).toContain('session_id');
        expect(columnNames).toContain('jti');
        expect(columnNames).toContain('refresh_token_hash');
        expect(columnNames).toContain('encrypted_replacement_token');
        expect(columnNames).toContain('expires_at');
        expect(columnNames).toContain('revoked_at');
        expect(columnNames).toContain('replaced_by_token_id');
        expect(columnNames).toContain('replaced_at');
        expect(columnNames).toContain('grace_expires_at');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('updated_at');
    });

    it('should have correct foreign keys referencing sessions and self-reference replaced_by_token_id', () => {
        const config = getTableConfig(tokens);
        expect(config.foreignKeys.length).toBe(2);

        const sessionFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'session_id'),
        );
        expect(sessionFk).toBeDefined();
        expect(sessionFk?.onDelete).toBe('cascade');

        const selfFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'replaced_by_token_id'),
        );
        expect(selfFk).toBeDefined();
        expect(selfFk?.onDelete).toBe('set null');
    });

    it('should have correct indexes', () => {
        const config = getTableConfig(tokens);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('tokens_jti_unique_idx');
        expect(indexNames).toContain('tokens_session_valid_idx');
        expect(indexNames).toContain('tokens_grace_idx');

        const uniqueIdx = indexes.find(idx => idx.config.name === 'tokens_jti_unique_idx');
        expect(uniqueIdx).toBeDefined();
        expect(uniqueIdx?.config.unique).toBe(true);
    });
});
