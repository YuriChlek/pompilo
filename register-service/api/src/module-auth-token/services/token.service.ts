import { Injectable } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { type StringValue } from 'ms';
import {
    AccessTokenPayload,
    RefreshTokenPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { getJwtSecrets } from '@/module-auth-token/utils/jwt-secret-rotation.util';

@Injectable()
export class TokenService {
    private readonly JWT_ACCESS_TOKEN_TTL: StringValue;
    private readonly JWT_REFRESH_TOKEN_TTL: StringValue;
    private readonly jwtSecrets: string[];

    public constructor(
        private readonly configService: ConfigService,
        private readonly jwtService: JwtService,
    ) {
        this.JWT_ACCESS_TOKEN_TTL =
            this.configService.getOrThrow<StringValue>('JWT_ACCESS_TOKEN_TTL');
        this.JWT_REFRESH_TOKEN_TTL =
            this.configService.getOrThrow<StringValue>('JWT_REFRESH_TOKEN_TTL');
        this.jwtSecrets = getJwtSecrets(this.configService);
    }

    createAccessToken(payload: AccessTokenPayload): string {
        return this.jwtService.sign(
            { ...payload },
            {
                expiresIn: this.JWT_ACCESS_TOKEN_TTL,
            },
        );
    }

    createRefreshToken(payload: RefreshTokenPayload): string {
        return this.jwtService.sign(
            { ...payload, type: 'refresh' },
            {
                expiresIn: this.JWT_REFRESH_TOKEN_TTL,
            },
        );
    }

    verifyToken(token: string): AccessTokenPayload | RefreshTokenPayload {
        let lastError: unknown;

        for (const secret of this.jwtSecrets) {
            try {
                return this.jwtService.verify(token, { secret });
            } catch (error) {
                lastError = error;
            }
        }

        throw lastError;
    }
}
