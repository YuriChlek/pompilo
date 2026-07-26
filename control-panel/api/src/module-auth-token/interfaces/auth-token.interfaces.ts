import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import type { TokenUserPayload } from '@/module-auth-token/types/token-db.types';
import { AuthRealm } from '@/module-auth/enums/auth.enums';

interface BaseTokenMeta {
    ipAddress: string;
    userAgent: string;
}

interface JwtPayloadBase<T extends TokenType> extends BaseTokenMeta {
    sub: string;
    userId: string;
    sessionId: string;
    type: T;
}

/** ===== DB ===== */
export interface TokenFromDbPayload extends BaseTokenMeta {
    tokenId: string;
    sessionId: string;
    jti: string;
    refreshToken: string;
    replacedByTokenId: string | null;
    replacedAt: Date | null;
    graceExpiresAt: Date | null;
    replacementSessionId: string | null;
    replacementRevokedAt: Date | null;
    user: TokenUserPayload;
    expiresAt: Date;
    revokedAt: Date | null;
    encryptedReplacementToken?: string | null;
}

/** ===== JWT ===== */
export interface RefreshTokenPayload extends JwtPayloadBase<TokenType.REFRESH> {
    jti: string;
    tokenId: string;
}

export interface AccessTokenPayload extends JwtPayloadBase<TokenType.ACCESS> {
    email: string;
    username: string;
    role: string | string[];
    realm: AuthRealm;
    jti: string;
}
