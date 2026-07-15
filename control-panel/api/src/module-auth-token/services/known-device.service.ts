import { ForbiddenException, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import ms, { type StringValue } from 'ms';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { TransactionRepository } from '@/module-drizzle/repository/transaction.repository';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { KnownDeviceSelect } from '@/module-auth-token/schemas/known-devices.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';
import type { KnownDeviceMetadata } from '@/module-auth-token/types/known-device.types';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';

@Injectable()
export class KnownDeviceService {
    public constructor(
        private readonly knownDeviceRepository: KnownDeviceRepository,
        private readonly sessionRepository: SessionRepository,
        private readonly securityEventRepository: SecurityEventRepository,
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly transactionRepository: TransactionRepository,
        private readonly redisTokenService: RedisTokenService,
        private readonly configService: ConfigService,
    ) {}

    async findKnownDevice(
        userId: string,
        realm: AuthRealm,
        deviceId: string,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect | null> {
        return await this.knownDeviceRepository.findActiveByDevice(
            userId,
            realm,
            deviceId,
            transaction,
        );
    }

    async findOrCreateKnownDevice(
        userId: string,
        realm: AuthRealm,
        deviceId: string,
        metadata?: KnownDeviceMetadata,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect> {
        const existing = await this.findKnownDevice(userId, realm, deviceId, transaction);
        if (existing) {
            return existing;
        }

        try {
            return await this.knownDeviceRepository.save(
                {
                    userId,
                    realm,
                    deviceId,
                    ...metadata,
                },
                transaction,
            );
        } catch (error) {
            const active = await this.findKnownDevice(userId, realm, deviceId, transaction);
            if (active) {
                return active;
            }
            throw error;
        }
    }

    async listKnownDevices(
        userId: string,
        realm: AuthRealm,
        transaction?: RepositoryTransaction,
    ): Promise<KnownDeviceSelect[]> {
        return await this.knownDeviceRepository.listActive(userId, realm, transaction);
    }

    async touchKnownDevice(
        knownDeviceId: string,
        metadata?: KnownDeviceMetadata,
        throttleWindowSeconds?: number,
        transaction?: RepositoryTransaction,
        now = new Date(),
    ): Promise<KnownDeviceSelect | null> {
        const device = await this.knownDeviceRepository.findById(knownDeviceId, transaction);
        if (!device || device.revokedAt) {
            return null;
        }

        const ipChanged =
            metadata?.lastIpAddress !== undefined &&
            metadata.lastIpAddress !== device.lastIpAddress;
        const countryChanged =
            metadata?.lastCountry !== undefined && metadata.lastCountry !== device.lastCountry;
        const regionChanged =
            metadata?.lastRegion !== undefined && metadata.lastRegion !== device.lastRegion;
        const cityChanged =
            metadata?.lastCity !== undefined && metadata.lastCity !== device.lastCity;
        const uaChanged =
            metadata?.lastUserAgent !== undefined &&
            metadata.lastUserAgent !== device.lastUserAgent;

        const metadataChanged =
            ipChanged || countryChanged || regionChanged || cityChanged || uaChanged;

        const timeDiffMs = now.getTime() - device.lastSeenAt.getTime();
        const shouldThrottle =
            !metadataChanged &&
            throttleWindowSeconds !== undefined &&
            timeDiffMs < throttleWindowSeconds * 1000;

        if (shouldThrottle) {
            return device;
        }

        return await this.knownDeviceRepository.update(
            knownDeviceId,
            {
                lastSeenAt: now,
                ...metadata,
            },
            transaction,
        );
    }

    async trustKnownDevice(
        knownDeviceId: string,
        trustExpiresAt?: Date | null,
        transaction?: RepositoryTransaction,
        now = new Date(),
    ): Promise<KnownDeviceSelect | null> {
        const device = await this.knownDeviceRepository.findById(knownDeviceId, transaction);
        if (!device || device.revokedAt) {
            return null;
        }

        return await this.knownDeviceRepository.update(
            knownDeviceId,
            {
                trustedAt: now,
                trustExpiresAt: trustExpiresAt ?? null,
            },
            transaction,
        );
    }

    async revokeKnownDevice(
        userId: string,
        realm: AuthRealm,
        knownDeviceId: string,
    ): Promise<void> {
        await this.transactionRepository.run(async transaction => {
            const device = await this.knownDeviceRepository.findByIdForUpdate(
                knownDeviceId,
                transaction,
            );
            if (!device) {
                throw new NotFoundException('Known device not found');
            }

            if (device.userId !== userId || device.realm !== realm) {
                throw new ForbiddenException('Access denied');
            }

            if (device.revokedAt) {
                return;
            }

            const affectedSessions =
                await this.sessionRepository.findUnrevokedByKnownDeviceForUpdate(
                    knownDeviceId,
                    transaction,
                );
            const sessionIds = affectedSessions.map(s => s.id);

            if (sessionIds.length > 0) {
                const accessTokenTtlSeconds = Math.ceil(
                    ms(this.configService.getOrThrow<StringValue>('JWT_ACCESS_TOKEN_TTL')) / 1000,
                );
                const clockSkewSeconds =
                    this.configService.getOrThrow<number>('AUTH_CLOCK_SKEW_SECONDS');
                const blacklistTtlSeconds = accessTokenTtlSeconds + clockSkewSeconds;

                for (const sessionId of sessionIds) {
                    await this.redisTokenService.revokeSession(sessionId, blacklistTtlSeconds);
                }
            }

            const now = new Date();
            await this.knownDeviceRepository.revoke(knownDeviceId, now, transaction);

            if (sessionIds.length > 0) {
                await this.sessionRepository.revokeUnrevokedByKnownDevice(
                    knownDeviceId,
                    now,
                    transaction,
                );

                for (const session of affectedSessions) {
                    await this.authTokenRepository.revokeTokensBySession(
                        session.id,
                        now,
                        transaction,
                    );
                    await this.securityEventRepository.save(
                        {
                            userId: session.userId,
                            realm: session.realm as AuthRealm,
                            sessionId: session.id,
                            eventType: SecurityEventType.SESSION_REVOKED,
                            ipAddress: session.ipAddress,
                            userAgent: session.userAgent,
                            metadata: {
                                revocationReason: 'known_device_revoked',
                            },
                        },
                        transaction,
                    );
                }
            }
        });
    }
}
