import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import ms, { type StringValue } from 'ms';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { RefreshTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class RefreshTokenStorageService {
    private readonly refreshTokenTtl: StringValue;

    constructor(
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly sessionRepository: SessionRepository,
        configService: ConfigService,
    ) {
        this.refreshTokenTtl = configService.getOrThrow<StringValue>('JWT_REFRESH_TOKEN_TTL');
    }

    async saveRefreshToken(
        refreshTokenPayload: RefreshTokenPayload,
        refreshToken: string,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const refreshTokenHash = await Argon2HashUtil.hash(refreshToken);
        const defaultExpiresAt = new Date(Date.now() + ms(this.refreshTokenTtl));

        let expiresAt = defaultExpiresAt;
        const session = await this.sessionRepository.findById(
            refreshTokenPayload.sessionId,
            transaction,
        );
        if (session && session.expiresAt) {
            if (session.expiresAt.getTime() < defaultExpiresAt.getTime()) {
                expiresAt = session.expiresAt;
            }
        }

        await this.authTokenRepository.createRefreshToken(
            refreshTokenPayload,
            refreshTokenHash,
            expiresAt,
            transaction,
        );
    }
}
