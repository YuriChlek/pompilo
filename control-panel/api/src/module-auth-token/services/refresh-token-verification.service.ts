import { BadRequestException, Injectable } from '@nestjs/common';
import {
    type AccessTokenPayload,
    RefreshTokenPayload,
    TokenFromDbPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { RefreshTokenVerificationResult } from '@/module-auth-token/types/auth-token.types';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { TokenService } from '@/module-auth-token/services/token.service';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { AuthRealm } from '@/module-auth/enums/auth.enums';

@Injectable()
export class RefreshTokenVerificationService {
    constructor(
        private readonly tokenService: TokenService,
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly sessionRepository: SessionRepository,
    ) {}

    async verify(
        token: string,
        expectedRealm?: AuthRealm,
    ): Promise<RefreshTokenVerificationResult> {
        const payload: AccessTokenPayload | RefreshTokenPayload =
            this.tokenService.verifyToken(token);

        if (!payload || payload.type !== TokenType.REFRESH || !('tokenId' in payload)) {
            throw new BadRequestException('Refresh token not found.');
        }

        const refreshTokenData: TokenFromDbPayload | null =
            await this.authTokenRepository.findRefreshTokenById(payload.tokenId);

        if (!refreshTokenData) {
            return { verified: false };
        }

        const { refreshToken, tokenId, sessionId, jti, user } = refreshTokenData;

        if (
            tokenId !== payload.tokenId ||
            sessionId !== payload.sessionId ||
            jti !== payload.jti ||
            this.isRevokedToken(refreshTokenData) ||
            this.isExpiredToken(refreshTokenData)
        ) {
            return { verified: false };
        }

        const isCurrent = this.isCurrentToken(refreshTokenData);
        const isGrace = this.isValidGraceToken(refreshTokenData);

        if (!isCurrent && !isGrace) {
            return { verified: false };
        }

        const verified = await Argon2HashUtil.compare(token, refreshToken);

        if (!verified) {
            return { verified };
        }

        const session = await this.sessionRepository.findById(sessionId);
        if (!session) {
            return { verified: false };
        }

        const now = new Date();
        if (session.revokedAt !== null || session.expiresAt.getTime() <= now.getTime()) {
            return { verified: false };
        }

        if (expectedRealm && session.realm !== expectedRealm) {
            return { verified: false };
        }

        return {
            verified,
            tokenId,
            sessionId,
            user,
            isGrace,
        };
    }

    public isCurrentToken(token: TokenFromDbPayload, now = new Date()): boolean {
        return (
            token.revokedAt === null &&
            token.replacedAt === null &&
            token.expiresAt.getTime() > now.getTime()
        );
    }

    public isValidGraceToken(token: TokenFromDbPayload, now = new Date()): boolean {
        return (
            token.revokedAt === null &&
            token.replacedAt !== null &&
            token.graceExpiresAt !== null &&
            token.graceExpiresAt.getTime() > now.getTime() &&
            token.replacementSessionId === token.sessionId &&
            token.replacementRevokedAt === null
        );
    }

    public isRevokedToken(token: TokenFromDbPayload): boolean {
        return token.revokedAt !== null;
    }

    public isExpiredToken(token: TokenFromDbPayload, now = new Date()): boolean {
        return token.expiresAt.getTime() <= now.getTime();
    }
}
