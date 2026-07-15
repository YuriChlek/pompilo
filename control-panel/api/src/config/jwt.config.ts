import { ConfigService } from '@nestjs/config';
import { JwtModuleOptions } from '@nestjs/jwt';

export const getJWTConfig = (configService: ConfigService): JwtModuleOptions => {
    return {
        secret: configService.get<string>('JWT_SECRET'),
        signOptions: {
            algorithm: 'HS256',
        },
        verifyOptions: {
            algorithms: ['HS256'],
            ignoreExpiration: false,
        },
    };
};
