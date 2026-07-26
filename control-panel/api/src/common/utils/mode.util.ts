import { ConfigService } from '@nestjs/config';

export const getMode = (configService: ConfigService): boolean =>
    configService.getOrThrow<string>('NODE_ENV') === 'development';
