import { ConfigService } from '@nestjs/config';
import { getDrizzleDbConfig } from '@/module-drizzle/db-config/drizzle-db.config';

describe('drizzleDbConfig', () => {
    const baseConfig = {
        DB_HOST: 'postgres',
        DB_PORT: '5432',
        DB_USER: 'app',
        DB_PASSWORD: 'secret',
        DB_NAME: 'identity_service',
    };

    it('applies the configured pool size below the per-replica cap', () => {
        const config = getDrizzleDbConfig(new ConfigService({ ...baseConfig, DB_POOL_MAX: '20' }));

        expect(config).toEqual({
            host: 'postgres',
            port: 5432,
            user: 'app',
            password: 'secret',
            database: 'identity_service',
            max: 20,
        });
    });

    it('caps the pool at 30 connections per API replica', () => {
        const config = getDrizzleDbConfig(new ConfigService({ ...baseConfig, DB_POOL_MAX: '100' }));

        expect(config.max).toBe(30);
    });
});
