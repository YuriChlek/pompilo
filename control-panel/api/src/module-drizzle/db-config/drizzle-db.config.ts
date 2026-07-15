import { ConfigService } from '@nestjs/config';
import { getDbPoolMax } from '@/config/capacity-limits.config';
import { DrizzleDbConfig } from '@/module-drizzle/interfaces/drizzle-db-config.interfaces';

export const getDrizzleDbConfig = (configService: ConfigService): DrizzleDbConfig => {
    return {
        host: configService.getOrThrow<string>('DB_HOST'),
        port: Number(configService.getOrThrow<string>('DB_PORT')),
        user: configService.getOrThrow<string>('DB_USER'),
        password: configService.getOrThrow<string>('DB_PASSWORD'),
        database: configService.getOrThrow<string>('DB_NAME'),
        max: getDbPoolMax(configService),
    };
};
