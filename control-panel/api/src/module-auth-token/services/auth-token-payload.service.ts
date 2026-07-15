import { Injectable } from '@nestjs/common';
import { randomBytes, randomUUID } from 'crypto';
import {
    type AccessTokenPayload,
    RefreshTokenPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { UserJwtPayload } from '@/module-user/interfaces/user.interfaces';
import { normalizeStr } from '@/common/utils/string.utils';
import { deriveAuthRealmFromRole, UserRoles } from '@/module-auth/enums/auth.enums';

@Injectable()
export class AuthTokenPayloadService {
    createAccessTokenPayload(user: UserJwtPayload): AccessTokenPayload {
        const role = user.role;
        return {
            sub: user.id,
            userId: user.id,
            sessionId: user.sessionId ?? '',
            email: user.email,
            username: user.name,
            role,
            realm: deriveAuthRealmFromRole(role),
            ipAddress: user.ipAddress,
            userAgent: normalizeStr(user.userAgent),
            type: TokenType.ACCESS,
            jti: randomBytes(64).toString('hex'),
        };
    }

    createRefreshTokenPayload(user: UserJwtPayload): RefreshTokenPayload {
        return {
            sub: user.id,
            userId: user.id,
            sessionId: user.sessionId ?? '',
            userAgent: normalizeStr(user.userAgent),
            ipAddress: user.ipAddress,
            type: TokenType.REFRESH,
            jti: randomBytes(64).toString('hex'),
            tokenId: randomUUID(),
        };
    }
}
