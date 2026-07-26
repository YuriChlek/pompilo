import { Provider } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Pool } from 'pg';
import { drizzle, NodePgDatabase } from 'drizzle-orm/node-postgres';
import { getDrizzleDbConfig } from '@/module-drizzle/db-config/drizzle-db.config';
import * as schema from '@/module-drizzle/schemas';

export const DRIZZLE_PROVIDER = Symbol.for('DRIZZLE_PROVIDER');

export const DrizzleProvider: Provider[] = [
    {
        provide: DRIZZLE_PROVIDER,
        inject: [ConfigService],
        useFactory: (configService: ConfigService) => {
            const dbConfig = getDrizzleDbConfig(configService);
            const pool = new Pool(dbConfig);
            return drizzle(pool, { schema }) as NodePgDatabase<typeof schema>;
        },
    },
];
