import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { Request, Response } from 'express';
import { randomUUID } from 'crypto';
import ms, { type StringValue } from 'ms';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import { UserJwtPayload } from '@/module-user/interfaces/user.interfaces';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { getRequestMetadata } from '@/common/utils/request-metadata.util';

@Injectable()
export class AuthSessionService {
    private readonly cookieDomain: StringValue;
    private readonly accessTokenTtl: StringValue;
    private readonly refreshTokenTtl: StringValue;

    constructor(
        private readonly authTokenService: AuthTokenService,
        private readonly configService: ConfigService,
    ) {
        this.cookieDomain = this.configService.getOrThrow<StringValue>('COOKIE_DOMAIN');
        this.accessTokenTtl = this.configService.getOrThrow<StringValue>('JWT_ACCESS_TOKEN_TTL');
        this.refreshTokenTtl = this.configService.getOrThrow<StringValue>('JWT_REFRESH_TOKEN_TTL');
    }

    async issueTokens(
        response: Response,
        payload: UserJwtPayload,
        setRefreshToken = true,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const { accessToken, refreshToken, payloadWithSession } = await this.issueTokensDeferred(
            payload,
            setRefreshToken,
            transaction,
        );

        this.applyTokens(response, {
            accessToken,
            refreshToken,
            role: payloadWithSession.role,
        });
    }

    async issueTokensDeferred(
        payload: UserJwtPayload,
        setRefreshToken = true,
        transaction?: RepositoryTransaction,
    ): Promise<{
        accessToken: string;
        refreshToken: string | null;
        payloadWithSession: UserJwtPayload;
    }> {
        try {
            if (!setRefreshToken && !payload.sessionId) {
                throw new Error('Session ID is required when refreshing access tokens');
            }

            const payloadWithSession: UserJwtPayload = {
                ...payload,
                sessionId: payload.sessionId ?? randomUUID(),
            };
            const accessToken = this.authTokenService.createAccessToken(payloadWithSession);
            const refreshToken: string | null = setRefreshToken
                ? await this.authTokenService.createInitialRefreshToken(
                      payloadWithSession.sessionId!,
                      payloadWithSession,
                      transaction,
                  )
                : null;

            return { accessToken, refreshToken, payloadWithSession };
        } catch (error) {
            throw new InternalServerErrorException(
                error instanceof Error ? error.message : 'Failed to set authentication tokens',
            );
        }
    }

    applyTokens(
        response: Response,
        tokens: {
            accessToken: string;
            refreshToken: string | null;
            role: UserRoles;
        },
    ): void {
        this.setTokenCookie(response, tokens.accessToken, TokenType.ACCESS, tokens.role);
        if (tokens.refreshToken) {
            this.setTokenCookie(response, tokens.refreshToken, TokenType.REFRESH, tokens.role);
        }
    }

    clearTokens(response: Response, userRole: UserRoles): void {
        this.removeTokenCookie(response, userRole, this.getCookieName(userRole, TokenType.ACCESS));
        this.removeTokenCookie(response, userRole, this.getCookieName(userRole, TokenType.REFRESH));
    }

    clearOppositeCustomerTokens(_response: Response, _currentUserRole: UserRoles): void {
        return;
    }

    getCookieName(userRole: UserRoles, type: TokenType): string {
        if (
            userRole === UserRoles.PLATFORM_ADMIN || userRole === UserRoles.SUPER_ADMIN
        ) {
            return type === TokenType.ACCESS
                ? COOKIE_NAMES.ADMIN_ACCESS_TOKEN
                : COOKIE_NAMES.ADMIN_REFRESH_TOKEN;
        }

        return type === TokenType.ACCESS
            ? COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN
            : COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN;
    }

    getUserMetaData(request: Request): { ipAddress: string; userAgent: string } {
        return getRequestMetadata(request);
    }

    private setTokenCookie(
        response: Response,
        token: string,
        type: TokenType,
        userRole: UserRoles,
    ): void {
        const ttl = type === TokenType.ACCESS ? this.accessTokenTtl : this.refreshTokenTtl;
        const path = this.getCookiePath(userRole);
        const isProduction = this.configService.get('NODE_ENV') === 'production';

        response.cookie(this.getCookieName(userRole, type), token, {
            httpOnly: true,
            secure: isProduction,
            sameSite: 'lax',
            domain: this.cookieDomain,
            expires: new Date(Date.now() + ms(ttl)),
            path,
        });
    }

    private removeTokenCookie(response: Response, userRole: UserRoles, tokenName: string): void {
        const path = this.getCookiePath(userRole);
        const isProduction = this.configService.get('NODE_ENV') === 'production';

        response.cookie(tokenName, '', {
            httpOnly: true,
            secure: isProduction,
            sameSite: 'lax',
            domain: this.cookieDomain,
            expires: new Date(0),
            path,
        });
    }

    private getCookiePath(userRole: UserRoles): string {
        // Authenticated browser requests are proxied through `/api/...` on the Next app.
        // Role-scoped cookie paths like `/admin` prevent those cookies from
        // reaching the proxy endpoints that forward API requests to the backend.
        void userRole;
        return '/';
    }
}
