import { defineConfig } from 'drizzle-kit';
import { loadApiEnvFiles } from '@config/api-env.config';

loadApiEnvFiles();

export default defineConfig({
    schema: './src/module-drizzle/schemas/index.ts',
    out: './drizzle-migrations',
    dialect: 'postgresql',
    dbCredentials: {
        host: process.env.DB_HOST!,
        port: Number(process.env.DB_PORT),
        user: process.env.DB_USER!,
        password: process.env.DB_PASSWORD!,
        database: process.env.DB_NAME!,
        ssl: false,
    },
});
