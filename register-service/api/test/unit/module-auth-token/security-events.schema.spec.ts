import { securityEvents } from '@/module-auth-token/schemas/security-events.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('SecurityEventsSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(securityEvents);
        expect(config.name).toBe('security_events');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(securityEvents);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('security_event_id');
        expect(columnNames).toContain('user_id');
        expect(columnNames).toContain('realm');
        expect(columnNames).toContain('session_id');
        expect(columnNames).toContain('known_device_id');
        expect(columnNames).toContain('event_type');
        expect(columnNames).toContain('risk_score');
        expect(columnNames).toContain('risk_reason');
        expect(columnNames).toContain('ip_address');
        expect(columnNames).toContain('country');
        expect(columnNames).toContain('region');
        expect(columnNames).toContain('city');
        expect(columnNames).toContain('user_agent');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('metadata');
    });

    it('should have correct foreign key referencing users, known_devices and sessions', () => {
        const config = getTableConfig(securityEvents);
        expect(config.foreignKeys.length).toBeGreaterThanOrEqual(3);

        const userFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'user_id'),
        );
        expect(userFk).toBeDefined();
        expect(userFk?.onDelete).toBe('set null');

        const deviceFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'known_device_id'),
        );
        expect(deviceFk).toBeDefined();
        expect(deviceFk?.onDelete).toBe('set null');

        const sessionFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'session_id'),
        );
        expect(sessionFk).toBeDefined();
        expect(sessionFk?.onDelete).toBe('set null');
    });

    it('should have correct indexes', () => {
        const config = getTableConfig(securityEvents);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('security_events_device_idx');
        expect(indexNames).toContain('security_events_user_occurred_idx');

        const deviceIdx = indexes.find(idx => idx.config.name === 'security_events_device_idx');
        expect(deviceIdx).toBeDefined();
        expect(deviceIdx?.config.unique).toBeFalsy();

        const userOccurredIdx = indexes.find(
            idx => idx.config.name === 'security_events_user_occurred_idx',
        );
        expect(userOccurredIdx).toBeDefined();
        expect(userOccurredIdx?.config.unique).toBeFalsy();
    });

    it('should restrict realm values at the database level', () => {
        const config = getTableConfig(securityEvents);
        expect(config.checks.map(check => check.name)).toContain('security_events_realm_check');
    });
});
