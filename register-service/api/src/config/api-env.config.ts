import * as dotenv from 'dotenv';

export const API_ENV_FILE_PATHS = ['.env', '.env.development'] as const;

export function loadApiEnvFiles(): void {
    for (const envFilePath of API_ENV_FILE_PATHS) {
        dotenv.config({
            path: envFilePath,
            override: false,
        });
    }
}
