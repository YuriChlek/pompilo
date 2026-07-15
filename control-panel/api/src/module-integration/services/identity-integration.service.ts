import { ForbiddenException, Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { JwtService } from '@nestjs/jwt';
import { randomUUID } from 'crypto';
import { UserRepository } from '@/module-user/repository/user.repository';
import { UserModel } from '@/module-user/types/user.types';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { IdentityOutboxRepository } from '@/module-integration/repository/identity-outbox.repository';
import {
    IdentityContext,
    IdentityEventEnvelope,
    IdentityEventType,
    TradingOnboardingTokenPayload,
} from '@/module-integration/interfaces/identity-events.interfaces';

const DEFAULT_ONBOARDING_TOKEN_TTL_SECONDS = 5 * 60;
const DEFAULT_TRADING_ENTITLEMENT_VERSION = 1;

@Injectable()
export class IdentityIntegrationService {
    constructor(
        private readonly userRepository: UserRepository,
        private readonly identityOutboxRepository: IdentityOutboxRepository,
        private readonly configService: ConfigService,
        private readonly jwtService: JwtService,
    ) {}

    async getIdentityContext(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<IdentityContext> {
        const membership = await this.userRepository.findPrimaryMembership(userId, transaction);
        if (!membership) {
            throw new ServiceUnavailableException('identity_membership_context_missing');
        }

        return {
            userId,
            tenantId: membership.tenantId,
            membershipRole: membership.role,
        };
    }

    async recordUserRegistered(
        user: Pick<UserModel, 'id' | 'email' | 'name' | 'role'>,
        tenantId: string,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        await this.recordEvent(
            'identity.v1.UserRegistered',
            user.id,
            tenantId,
            {
                userId: user.id,
                tenantId,
                email: user.email,
                name: user.name,
                platformRole: user.role,
            },
            transaction,
        );
    }

    async recordEmailVerified(
        user: Pick<UserModel, 'id' | 'email' | 'name' | 'role'>,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const context = await this.getIdentityContext(user.id, transaction);
        await this.recordEvent(
            'identity.v1.EmailVerified',
            user.id,
            context.tenantId,
            {
                userId: user.id,
                tenantId: context.tenantId,
                email: user.email,
                verifiedAt: new Date().toISOString(),
            },
            transaction,
        );
        await this.recordTradingAccessGranted(user, context.tenantId, transaction);
    }

    async recordTradingAccessGranted(
        user: Pick<UserModel, 'id' | 'role'>,
        tenantId: string,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        await this.recordEvent(
            'identity.v1.TradingAccessGranted',
            user.id,
            tenantId,
            {
                userId: user.id,
                tenantId,
                entitlementVersion: DEFAULT_TRADING_ENTITLEMENT_VERSION,
                allowedExchanges: [],
                allowedStrategies: [],
                maxExchangeAccounts: 1,
                maxSymbols: 0,
                liveTradingAllowed: false,
                platformRole: user.role,
            },
            transaction,
        );
    }

    async recordUserDisabled(
        userId: string,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const context = await this.getIdentityContext(userId, transaction);
        await this.recordEvent(
            'identity.v1.UserDisabled',
            userId,
            context.tenantId,
            {
                userId,
                tenantId: context.tenantId,
                disabledAt: new Date().toISOString(),
            },
            transaction,
        );
    }

    async recordUserDeleted(userId: string, transaction: RepositoryTransaction): Promise<void> {
        const context = await this.getIdentityContext(userId, transaction);
        await this.recordEvent(
            'identity.v1.UserDeleted',
            userId,
            context.tenantId,
            {
                userId,
                tenantId: context.tenantId,
                deletedAt: new Date().toISOString(),
            },
            transaction,
        );
    }

    async createTradingOnboardingToken(userId: string): Promise<{
        token: string;
        expiresInSeconds: number;
        tokenType: 'Bearer';
    }> {
        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new ForbiddenException('identity_user_not_found');
        }
        if (!user.emailVerifiedAt) {
            throw new ForbiddenException('email_verification_required');
        }

        const context = await this.getIdentityContext(user.id);
        const secret = this.getOnboardingTokenSecret();
        const expiresInSeconds = this.getOnboardingTokenTtlSeconds();
        const payload: TradingOnboardingTokenPayload = {
            iss: 'identity-service',
            aud: 'trading-service',
            scope: 'trading:onboarding',
            sub: user.id,
            userId: user.id,
            tenantId: context.tenantId,
            membershipRole: context.membershipRole,
            emailVerified: true,
        };

        return {
            token: await this.jwtService.signAsync(payload, {
                secret,
                expiresIn: expiresInSeconds,
                jwtid: randomUUID(),
            }),
            expiresInSeconds,
            tokenType: 'Bearer',
        };
    }

    private async recordEvent(
        eventType: IdentityEventType,
        aggregateId: string,
        tenantId: string,
        payload: Record<string, unknown>,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const eventId = randomUUID();
        const eventVersion = 1;
        const occurredAt = new Date().toISOString();
        const envelope: IdentityEventEnvelope = {
            eventId,
            eventType,
            eventVersion,
            occurredAt,
            aggregateId,
            tenantId,
            payload,
        };

        await this.identityOutboxRepository.create(
            {
                eventId,
                eventType,
                eventVersion,
                aggregateId,
                tenantId,
                idempotencyKey: `${eventType}:${aggregateId}:${tenantId}:${eventVersion}`,
                payload: envelope,
                availableAt: new Date(occurredAt),
            },
            transaction,
        );
    }

    private getOnboardingTokenSecret(): string {
        return (
            this.configService.get<string>('TRADING_ONBOARDING_TOKEN_SECRET') ||
            this.configService.get<string>('JWT_SECRET') ||
            ''
        );
    }

    private getOnboardingTokenTtlSeconds(): number {
        const raw = this.configService.get<string | number>('TRADING_ONBOARDING_TOKEN_TTL_SECONDS');
        const parsed = typeof raw === 'number' ? raw : raw ? Number(raw) : NaN;

        return Number.isFinite(parsed) && parsed > 0
            ? parsed
            : DEFAULT_ONBOARDING_TOKEN_TTL_SECONDS;
    }
}
