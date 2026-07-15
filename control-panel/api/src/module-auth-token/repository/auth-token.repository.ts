import { Inject, Injectable } from '@nestjs/common';
import { and, eq, gt, lt, isNull, or, asc, isNotNull, notInArray, inArray } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { alias } from 'drizzle-orm/pg-core';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import * as schema from '@/module-drizzle/schemas';
import {
    RefreshTokenPayload,
    TokenFromDbPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { tokens } from '@/module-auth-token/schemas';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';
import { users } from '@/module-user/schemas';
import { NewTokenModel, TokenModel } from '@/module-auth-token/types/token-db.types';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class AuthTokenRepository {
    public constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async deleteExpiredTokens(
        retentionPeriodMs = 0,
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const now = new Date();
        const retentionThreshold = new Date(now.getTime() - retentionPeriodMs);

        // 1. Subquery for active (current) tokens that must not be deleted
        const activeTokens = dbClient
            .select({ id: tokens.id })
            .from(tokens)
            .where(and(isNull(tokens.revokedAt), gt(tokens.expiresAt, now)));

        // 2. Subquery for tokens currently in grace window
        const graceTokens = dbClient
            .select({ id: tokens.id })
            .from(tokens)
            .where(and(isNull(tokens.revokedAt), gt(tokens.graceExpiresAt, now)));

        // 3. Subquery for replacement tokens of grace window tokens
        const replacementTokens = dbClient
            .select({ id: tokens.replacedByTokenId })
            .from(tokens)
            .where(
                and(
                    isNull(tokens.revokedAt),
                    gt(tokens.graceExpiresAt, now),
                    isNotNull(tokens.replacedByTokenId),
                ),
            );

        // 4. Get the oldest eligible tokens (batch limit of 1000)
        // A token is eligible for cleanup if it is NOT active, NOT in grace window, NOT a replacement for a grace token,
        // and its expiresAt (or revokedAt if revoked) is older than the retention threshold.
        const eligibleTokensSubquery = dbClient
            .select({ id: tokens.id })
            .from(tokens)
            .where(
                and(
                    notInArray(tokens.id, activeTokens),
                    notInArray(tokens.id, graceTokens),
                    notInArray(tokens.id, replacementTokens),
                    or(
                        and(isNull(tokens.revokedAt), lt(tokens.expiresAt, retentionThreshold)),
                        and(isNotNull(tokens.revokedAt), lt(tokens.revokedAt, retentionThreshold)),
                    ),
                ),
            )
            .orderBy(asc(tokens.createdAt))
            .limit(1000);

        const result = await dbClient
            .delete(tokens)
            .where(inArray(tokens.id, eligibleTokensSubquery));

        return result.rowCount ?? 0;
    }

    async findById(
        tokenId: string,
        transaction?: RepositoryTransaction,
    ): Promise<TokenModel | null> {
        const dbClient = getTransactionClient(transaction, this.db);
        const [token] = await dbClient.select().from(tokens).where(eq(tokens.id, tokenId)).limit(1);
        return token || null;
    }

    async rotateToken(
        oldTokenId: string,
        newToken: NewTokenModel,
        graceExpiresAt: Date,
        encryptedReplacementToken?: string | null,
        transaction?: RepositoryTransaction,
    ): Promise<TokenModel | null> {
        if (transaction) {
            return this.rotateTokenWithClient(
                getTransactionClient(transaction, this.db),
                oldTokenId,
                newToken,
                graceExpiresAt,
                encryptedReplacementToken,
            );
        }

        return this.db.transaction(async tx =>
            this.rotateTokenWithClient(
                tx as unknown as NodePgDatabase<typeof schema>,
                oldTokenId,
                newToken,
                graceExpiresAt,
                encryptedReplacementToken,
            ),
        );
    }

    private async rotateTokenWithClient(
        dbClient: NodePgDatabase<typeof schema>,
        oldTokenId: string,
        newToken: NewTokenModel,
        graceExpiresAt: Date,
        encryptedReplacementToken?: string | null,
    ): Promise<TokenModel | null> {
        const now = new Date();
        const [insertedToken] = await dbClient.insert(tokens).values(newToken).returning();
        const [claimedToken] = await dbClient
            .update(tokens)
            .set({
                replacedByTokenId: insertedToken.id,
                replacedAt: now,
                graceExpiresAt,
                encryptedReplacementToken: encryptedReplacementToken ?? null,
                updatedAt: now,
            })
            .where(
                and(
                    eq(tokens.id, oldTokenId),
                    eq(tokens.sessionId, newToken.sessionId),
                    isNull(tokens.revokedAt),
                    isNull(tokens.replacedAt),
                    gt(tokens.expiresAt, now),
                ),
            )
            .returning();

        if (!claimedToken) {
            await dbClient.delete(tokens).where(eq(tokens.id, insertedToken.id));
            return null;
        }

        return insertedToken;
    }

    async createRefreshToken(
        refreshTokenPayload: RefreshTokenPayload,
        refreshTokenHash: string,
        expiresAt: Date,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const { sessionId, jti, tokenId } = refreshTokenPayload;
        const tokenToCreate: NewTokenModel = {
            id: tokenId,
            sessionId,
            jti,
            refreshTokenHash,
            expiresAt,
        };

        const dbClient = getTransactionClient(transaction, this.db);
        await dbClient.insert(tokens).values(tokenToCreate);
    }

    async findRefreshTokenById(tokenId: string): Promise<TokenFromDbPayload | null> {
        const replacementTokens = alias(tokens, 'replacement_tokens');
        const baseCondition = and(
            eq(tokens.id, tokenId),
            isNull(sessions.revokedAt),
            gt(sessions.expiresAt, new Date()),
        );

        const [refreshTokenData] = await this.db
            .select({
                tokenId: tokens.id,
                sessionId: tokens.sessionId,
                jti: tokens.jti,
                refreshTokenHash: tokens.refreshTokenHash,
                replacedByTokenId: tokens.replacedByTokenId,
                replacedAt: tokens.replacedAt,
                graceExpiresAt: tokens.graceExpiresAt,
                replacementSessionId: replacementTokens.sessionId,
                replacementRevokedAt: replacementTokens.revokedAt,
                ipAddress: sessions.ipAddress,
                userAgent: sessions.userAgent,
                userId: users.id,
                userName: users.name,
                userEmail: users.email,
                userRole: users.role,
                expiresAt: tokens.expiresAt,
                revokedAt: tokens.revokedAt,
                encryptedReplacementToken: tokens.encryptedReplacementToken,
            })
            .from(tokens)
            .innerJoin(sessions, eq(tokens.sessionId, sessions.id))
            .innerJoin(users, eq(sessions.userId, users.id))
            .leftJoin(replacementTokens, eq(tokens.replacedByTokenId, replacementTokens.id))
            .where(baseCondition)
            .limit(1);

        if (!refreshTokenData) {
            return null;
        }

        return {
            tokenId: refreshTokenData.tokenId,
            sessionId: refreshTokenData.sessionId,
            jti: refreshTokenData.jti,
            refreshToken: refreshTokenData.refreshTokenHash,
            replacedByTokenId: refreshTokenData.replacedByTokenId,
            replacedAt: refreshTokenData.replacedAt,
            graceExpiresAt: refreshTokenData.graceExpiresAt,
            replacementSessionId: refreshTokenData.replacementSessionId,
            replacementRevokedAt: refreshTokenData.replacementRevokedAt,
            ipAddress: refreshTokenData.ipAddress ?? '',
            userAgent: refreshTokenData.userAgent ?? '',
            user: {
                id: refreshTokenData.userId,
                name: refreshTokenData.userName,
                email: refreshTokenData.userEmail,
                role: refreshTokenData.userRole,
            },
            expiresAt: refreshTokenData.expiresAt,
            revokedAt: refreshTokenData.revokedAt,
            encryptedReplacementToken: refreshTokenData.encryptedReplacementToken,
        };
    }

    async revokeTokensBySession(
        sessionId: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<number> {
        const dbClient = getTransactionClient(transaction, this.db);
        const result = await dbClient
            .update(tokens)
            .set({ revokedAt: now, updatedAt: now })
            .where(and(eq(tokens.sessionId, sessionId), isNull(tokens.revokedAt)))
            .returning();
        return result.length;
    }

    async removeRefreshToken(sessionId: string): Promise<void> {
        await this.db
            .delete(tokens)
            .where(and(eq(tokens.sessionId, sessionId), isNull(tokens.revokedAt)));
    }
}
