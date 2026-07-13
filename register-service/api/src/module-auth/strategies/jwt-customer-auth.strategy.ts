import { PassportStrategy } from '@nestjs/passport';
import { Strategy, ExtractJwt } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { Request } from 'express';
import { JWT_ALGORITHM } from '@/module-auth/constants/auth.constants';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import {
    getJwtSecrets,
    selectJwtSecretForToken,
} from '@/module-auth-token/utils/jwt-secret-rotation.util';

@Injectable()
export class JwtCustomerAuthStrategy extends PassportStrategy(Strategy, 'customer-jwt') {
    constructor(configService: ConfigService) {
        const jwtSecrets = getJwtSecrets(configService);

        super({
            jwtFromRequest: ExtractJwt.fromExtractors([
                (request: Request): string | null => {
                    const cookiesKeys: string[] = request.cookies
                        ? Object.keys(request.cookies)
                        : [];

                    if (cookiesKeys.includes(COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN)) {
                        return request.cookies[COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN] as string;
                    }

                    return null;
                },
            ]),
            ignoreExpiration: false,
            secretOrKeyProvider: (_request: Request, rawJwtToken: string, done) => {
                done(null, selectJwtSecretForToken(rawJwtToken, jwtSecrets));
            },
            jsonWebTokenOptions: {
                maxAge: configService.getOrThrow<string>('JWT_ACCESS_TOKEN_TTL'),
            },
            algorithms: [JWT_ALGORITHM],
        });
    }

    validate(accessTokenPayload: AccessTokenPayload): AccessTokenPayload | null {
        if (
            accessTokenPayload.realm !== 'customer' ||
            accessTokenPayload.role !== UserRoles.USER
        ) {
            throw new UnauthorizedException('Invalid realm or role for customer strategy.');
        }

        return accessTokenPayload;
    }
}
