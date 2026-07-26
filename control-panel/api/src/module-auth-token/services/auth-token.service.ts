import { HttpException, Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import ms, { type StringValue } from 'ms';
import { TokenService } from '@/module-auth-token/services/token.service';
import {
    type AccessTokenPayload,
    RefreshTokenPayload,
    TokenFromDbPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { RefreshTokenVerificationResult } from '@/module-auth-token/types/auth-token.types';
import { UserJwtPayload } from '@/module-user/interfaces/user.interfaces';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { AuthTokenPayloadService } from '@/module-auth-token/services/auth-token-payload.service';
import { RefreshTokenVerificationService } from '@/module-auth-token/services/refresh-token-verification.service';
import { RefreshTokenStorageService } from '@/module-auth-token/services/refresh-token-storage.service';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { EncryptService } from '@/module-encrypt/services/encrypt.service';
import { AuthRealm } from '@/module-auth/enums/auth.enums';

@Injectable()
export class AuthTokenService {
    public constructor(
        private readonly tokenService: TokenService,
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly authTokenPayloadService: AuthTokenPayloadService,
        private readonly refreshTokenVerificationService: RefreshTokenVerificationService,
        private readonly refreshTokenStorageService: RefreshTokenStorageService,
        private readonly redisTokenService: RedisTokenService,
        private readonly configService: ConfigService,
        private readonly encryptService: EncryptService,
    ) {}

    createAccessToken(user: UserJwtPayload): string {
        try {
            const accessTokenPayload: AccessTokenPayload =
                this.authTokenPayloadService.createAccessTokenPayload(user);

            return this.tokenService.createAccessToken(accessTokenPayload);
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to create access token');
        }
    }

    getTokenData(token: string): AccessTokenPayload | RefreshTokenPayload {
        return this.tokenService.verifyToken(token);
    }

    async createRefreshToken(user: UserJwtPayload): Promise<string> {
        try {
            const refreshTokenPayload: RefreshTokenPayload =
                this.authTokenPayloadService.createRefreshTokenPayload(user);

            const refreshToken: string = this.tokenService.createRefreshToken(refreshTokenPayload);
            await this.refreshTokenStorageService.saveRefreshToken(
                refreshTokenPayload,
                refreshToken,
            );

            return refreshToken;
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to create refresh token');
        }
    }

    async createInitialRefreshToken(
        sessionId: string,
        user: UserJwtPayload,
        transaction?: RepositoryTransaction,
    ): Promise<string> {
        try {
            const userWithSession = { ...user, sessionId };
            const refreshTokenPayload: RefreshTokenPayload =
                this.authTokenPayloadService.createRefreshTokenPayload(userWithSession);

            const refreshToken: string = this.tokenService.createRefreshToken(refreshTokenPayload);
            await this.refreshTokenStorageService.saveRefreshToken(
                refreshTokenPayload,
                refreshToken,
                transaction,
            );

            return refreshToken;
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to create initial refresh token');
        }
    }

    async rotateRefreshToken(
        tokenRow: TokenFromDbPayload,
        session: SessionSelect,
        transaction?: RepositoryTransaction,
    ): Promise<string | null> {
        try {
            const user: UserJwtPayload = {
                id: tokenRow.user.id,
                name: tokenRow.user.name,
                email: tokenRow.user.email,
                role: tokenRow.user.role,
                sessionId: session.id,
                ipAddress: session.ipAddress ?? '',
                userAgent: session.userAgent ?? '',
            };

            const replacementPayload: RefreshTokenPayload =
                this.authTokenPayloadService.createRefreshTokenPayload(user);
            const rawRefreshToken = this.tokenService.createRefreshToken(replacementPayload);
            const refreshTokenHash = await Argon2HashUtil.hash(rawRefreshToken);
            const encryptedReplacementToken = this.encryptService.encrypt(rawRefreshToken);

            const now = new Date();
            const defaultExpiresAt = new Date(
                now.getTime() +
                    ms(this.configService.getOrThrow<StringValue>('JWT_REFRESH_TOKEN_TTL')),
            );
            let expiresAt = defaultExpiresAt;
            if (session.expiresAt && session.expiresAt.getTime() < defaultExpiresAt.getTime()) {
                expiresAt = session.expiresAt;
            }

            const graceExpiresAt = new Date(now.getTime() + 20_000);

            const rotated = await this.authTokenRepository.rotateToken(
                tokenRow.tokenId,
                {
                    id: replacementPayload.tokenId,
                    sessionId: session.id,
                    jti: replacementPayload.jti,
                    refreshTokenHash,
                    expiresAt,
                },
                graceExpiresAt,
                encryptedReplacementToken,
                transaction,
            );

            if (!rotated) {
                return null;
            }

            return rawRefreshToken;
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to rotate refresh token');
        }
    }

    async rotateRefreshTokenById(
        tokenId: string,
        session: SessionSelect,
        transaction?: RepositoryTransaction,
    ): Promise<string | null> {
        try {
            const tokenRow = await this.authTokenRepository.findRefreshTokenById(tokenId);
            if (!tokenRow) {
                return null;
            }
            return await this.rotateRefreshToken(tokenRow, session, transaction);
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to rotate refresh token by ID');
        }
    }

    async acceptReplacedTokenInGrace(tokenRow: TokenFromDbPayload): Promise<string | null> {
        try {
            await Promise.resolve();
            const now = new Date();
            const isGrace = this.refreshTokenVerificationService.isValidGraceToken(tokenRow, now);
            if (!isGrace) {
                return null;
            }

            if (!tokenRow.encryptedReplacementToken) {
                return null;
            }

            return this.encryptService.decrypt(tokenRow.encryptedReplacementToken);
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to accept replaced token in grace');
        }
    }

    async acceptReplacedTokenInGraceById(tokenId: string): Promise<string | null> {
        try {
            const tokenRow = await this.authTokenRepository.findRefreshTokenById(tokenId);
            if (!tokenRow) {
                return null;
            }

            return await this.acceptReplacedTokenInGrace(tokenRow);
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to accept replaced token in grace by ID');
        }
    }

    async verifyRefreshToken(
        token: string,
        expectedRealm?: AuthRealm,
    ): Promise<RefreshTokenVerificationResult> {
        return await this.refreshTokenVerificationService.verify(token, expectedRealm);
    }

    async removeRefreshToken(sessionId: string) {
        await this.authTokenRepository.removeRefreshToken(sessionId);
    }

    async revokeTokensBySession(sessionId: string, transaction?: RepositoryTransaction) {
        await this.authTokenRepository.revokeTokensBySession(sessionId, new Date(), transaction);
    }

    async revokeAccessSession(payload: AccessTokenPayload): Promise<void> {
        const ttlSeconds = Math.ceil(
            ms(this.configService.getOrThrow<StringValue>('JWT_ACCESS_TOKEN_TTL')) / 1000,
        );

        if (payload.jti) {
            await this.redisTokenService.revokeToken(payload.jti, ttlSeconds);
        }

        if (payload.sessionId) {
            await this.redisTokenService.revokeSession(payload.sessionId, ttlSeconds);
        }
    }

    private handleUnexpectedError(error: unknown, message: string): never {
        if (error instanceof HttpException) {
            throw error;
        }

        throw new InternalServerErrorException(message);
    }
}
