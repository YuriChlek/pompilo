import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { KnownDeviceSelect } from '@/module-auth-token/schemas/known-devices.schema';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import {
    TransactionRepository,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { ConfigService } from '@nestjs/config';
import { ForbiddenException, NotFoundException } from '@nestjs/common';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';

const buildMockDevice = (overrides: Partial<KnownDeviceSelect> = {}): KnownDeviceSelect => ({
    id: overrides.id ?? 'known-device-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    deviceId: overrides.deviceId ?? 'device-uuid',
    trustedAt: overrides.trustedAt ?? null,
    trustExpiresAt: overrides.trustExpiresAt ?? null,
    revokedAt: overrides.revokedAt ?? null,
    firstSeenAt: overrides.firstSeenAt ?? new Date(),
    lastSeenAt: overrides.lastSeenAt ?? new Date(),
    lastIpAddress: overrides.lastIpAddress ?? null,
    lastCountry: overrides.lastCountry ?? null,
    lastRegion: overrides.lastRegion ?? null,
    lastCity: overrides.lastCity ?? null,
    lastUserAgent: overrides.lastUserAgent ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
});

const buildMockSession = (overrides: Partial<SessionSelect> = {}): SessionSelect => ({
    id: overrides.id ?? 'session-id',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    knownDeviceId: overrides.knownDeviceId ?? 'known-device-uuid',
    deviceId: overrides.deviceId ?? 'device-uuid',
    ipAddress: overrides.ipAddress ?? null,
    userAgent: overrides.userAgent ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
    lastSeenAt: overrides.lastSeenAt ?? new Date(),
    expiresAt: overrides.expiresAt ?? new Date(),
    revokedAt: overrides.revokedAt ?? null,
    lastCountry: overrides.lastCountry ?? null,
    lastRegion: overrides.lastRegion ?? null,
    lastCity: overrides.lastCity ?? null,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
});

describe('KnownDeviceService', () => {
    let service: KnownDeviceService;
    let repository: {
        findActiveByDevice: jest.MockedFunction<KnownDeviceRepository['findActiveByDevice']>;
        save: jest.MockedFunction<KnownDeviceRepository['save']>;
        listActive: jest.MockedFunction<KnownDeviceRepository['listActive']>;
        findById: jest.MockedFunction<KnownDeviceRepository['findById']>;
        findByIdForUpdate: jest.MockedFunction<KnownDeviceRepository['findByIdForUpdate']>;
        update: jest.MockedFunction<KnownDeviceRepository['update']>;
        revoke: jest.MockedFunction<KnownDeviceRepository['revoke']>;
    };
    let sessionRepository: {
        findUnrevokedByKnownDeviceForUpdate: jest.MockedFunction<
            SessionRepository['findUnrevokedByKnownDeviceForUpdate']
        >;
        revokeUnrevokedByKnownDevice: jest.MockedFunction<
            SessionRepository['revokeUnrevokedByKnownDevice']
        >;
    };
    let securityEventRepository: {
        save: jest.MockedFunction<SecurityEventRepository['save']>;
    };
    let authTokenRepository: {
        revokeTokensBySession: jest.MockedFunction<AuthTokenRepository['revokeTokensBySession']>;
    };
    let transactionRepository: {
        run: jest.MockedFunction<TransactionRepository['run']>;
    };
    let redisTokenService: {
        revokeSession: jest.MockedFunction<RedisTokenService['revokeSession']>;
    };
    let configService: {
        getOrThrow: jest.MockedFunction<ConfigService['getOrThrow']>;
    };
    beforeEach(() => {
        repository = {
            findActiveByDevice: jest.fn(),
            save: jest.fn(),
            listActive: jest.fn(),
            findById: jest.fn(),
            findByIdForUpdate: jest.fn(),
            update: jest.fn(),
            revoke: jest.fn().mockResolvedValue(true),
        };
        sessionRepository = {
            findUnrevokedByKnownDeviceForUpdate: jest.fn(),
            revokeUnrevokedByKnownDevice: jest.fn().mockResolvedValue(0),
        };
        securityEventRepository = {
            save: jest.fn(),
        };
        authTokenRepository = {
            revokeTokensBySession: jest.fn().mockResolvedValue(1),
        };
        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation(
                    async <T>(work: (tx: RepositoryTransaction) => Promise<T>): Promise<T> => {
                        return await work({} as RepositoryTransaction);
                    },
                ),
        };
        redisTokenService = {
            revokeSession: jest.fn(),
        };
        configService = {
            getOrThrow: jest.fn().mockImplementation((key: string) => {
                if (key === 'JWT_ACCESS_TOKEN_TTL') return '15m';
                if (key === 'AUTH_CLOCK_SKEW_SECONDS') return 60;
                return undefined;
            }),
        };

        service = new KnownDeviceService(
            repository as unknown as KnownDeviceRepository,
            sessionRepository as unknown as SessionRepository,
            securityEventRepository as unknown as SecurityEventRepository,
            authTokenRepository as unknown as AuthTokenRepository,
            transactionRepository as unknown as TransactionRepository,
            redisTokenService as unknown as RedisTokenService,
            configService as unknown as ConfigService,
        );
    });

    describe('findKnownDevice', () => {
        it('should return found active device', async () => {
            const device = buildMockDevice();
            repository.findActiveByDevice.mockResolvedValue(device);

            const result = await service.findKnownDevice('user-uuid', 'customer', 'device-uuid');

            expect(repository.findActiveByDevice).toHaveBeenCalledWith(
                'user-uuid',
                'customer',
                'device-uuid',
                undefined,
            );
            expect(result).toEqual(device);
        });

        it('should return null if active device not found', async () => {
            repository.findActiveByDevice.mockResolvedValue(null);

            const result = await service.findKnownDevice('user-uuid', 'customer', 'device-uuid');

            expect(result).toBeNull();
        });
    });

    describe('findOrCreateKnownDevice', () => {
        it('should return existing device without calling save if already active', async () => {
            const device = buildMockDevice();
            repository.findActiveByDevice.mockResolvedValue(device);

            const result = await service.findOrCreateKnownDevice(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(repository.findActiveByDevice).toHaveBeenCalledTimes(1);
            expect(repository.save).not.toHaveBeenCalled();
            expect(result).toEqual(device);
        });

        it('should save and return new device if not exists', async () => {
            const device = buildMockDevice();
            repository.findActiveByDevice.mockResolvedValue(null);
            repository.save.mockResolvedValue(device);

            const result = await service.findOrCreateKnownDevice(
                'user-uuid',
                'customer',
                'device-uuid',
                { lastIpAddress: '1.2.3.4' },
            );

            expect(repository.findActiveByDevice).toHaveBeenCalledTimes(1);
            expect(repository.save).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    deviceId: 'device-uuid',
                    lastIpAddress: '1.2.3.4',
                },
                undefined,
            );
            expect(result).toEqual(device);
        });

        it('should handle unique constraint race by returning existing active device', async () => {
            const device = buildMockDevice();
            repository.findActiveByDevice
                .mockResolvedValueOnce(null) // first check: not found
                .mockResolvedValueOnce(device); // second check after error: found

            repository.save.mockRejectedValue(new Error('Unique constraint violation'));

            const result = await service.findOrCreateKnownDevice(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(repository.findActiveByDevice).toHaveBeenCalledTimes(2);
            expect(repository.save).toHaveBeenCalled();
            expect(result).toEqual(device);
        });

        it('should rethrow error if unique constraint race occurs but no active device is found', async () => {
            repository.findActiveByDevice.mockResolvedValue(null);
            repository.save.mockRejectedValue(new Error('Some DB error'));

            await expect(
                service.findOrCreateKnownDevice('user-uuid', 'customer', 'device-uuid'),
            ).rejects.toThrow('Some DB error');

            expect(repository.findActiveByDevice).toHaveBeenCalledTimes(2);
            expect(repository.save).toHaveBeenCalled();
        });
    });

    describe('listKnownDevices', () => {
        it('should return list of active devices', async () => {
            const devices = [buildMockDevice({ id: '1' }), buildMockDevice({ id: '2' })];
            repository.listActive.mockResolvedValue(devices);

            const result = await service.listKnownDevices('user-uuid', 'customer');

            expect(repository.listActive).toHaveBeenCalledWith('user-uuid', 'customer', undefined);
            expect(result).toEqual(devices);
        });
    });

    describe('touchKnownDevice', () => {
        it('should return null if device not found', async () => {
            repository.findById.mockResolvedValue(null);

            const result = await service.touchKnownDevice('device-id');

            expect(result).toBeNull();
            expect(repository.update).not.toHaveBeenCalled();
        });

        it('should return null if device is revoked', async () => {
            const revokedDevice = buildMockDevice({ revokedAt: new Date() });
            repository.findById.mockResolvedValue(revokedDevice);

            const result = await service.touchKnownDevice('device-id');

            expect(result).toBeNull();
            expect(repository.update).not.toHaveBeenCalled();
        });

        it('should throttle and return existing device if metadata is unchanged and time diff is within throttle window', async () => {
            const lastSeen = new Date('2026-06-23T12:00:00Z');
            const now = new Date('2026-06-23T12:01:00Z'); // 60s later
            const device = buildMockDevice({
                lastSeenAt: lastSeen,
                lastIpAddress: '1.2.3.4',
            });
            repository.findById.mockResolvedValue(device);

            const result = await service.touchKnownDevice(
                'device-id',
                { lastIpAddress: '1.2.3.4' },
                120, // 120s throttle window
                undefined,
                now,
            );

            expect(result).toEqual(device);
            expect(repository.update).not.toHaveBeenCalled();
        });

        it('should update and not throttle if metadata has changed', async () => {
            const lastSeen = new Date('2026-06-23T12:00:00Z');
            const now = new Date('2026-06-23T12:01:00Z');
            const device = buildMockDevice({
                lastSeenAt: lastSeen,
                lastIpAddress: '1.2.3.4',
            });
            const updatedDevice = buildMockDevice({
                lastSeenAt: now,
                lastIpAddress: '5.6.7.8',
            });
            repository.findById.mockResolvedValue(device);
            repository.update.mockResolvedValue(updatedDevice);

            const result = await service.touchKnownDevice(
                'device-id',
                { lastIpAddress: '5.6.7.8' },
                120,
                undefined,
                now,
            );

            expect(result).toEqual(updatedDevice);
            expect(repository.update).toHaveBeenCalledWith(
                'device-id',
                {
                    lastSeenAt: now,
                    lastIpAddress: '5.6.7.8',
                },
                undefined,
            );
        });

        it('should update and not throttle if time diff is outside throttle window', async () => {
            const lastSeen = new Date('2026-06-23T12:00:00Z');
            const now = new Date('2026-06-23T12:03:00Z'); // 180s later
            const device = buildMockDevice({
                lastSeenAt: lastSeen,
                lastIpAddress: '1.2.3.4',
            });
            const updatedDevice = buildMockDevice({
                lastSeenAt: now,
                lastIpAddress: '1.2.3.4',
            });
            repository.findById.mockResolvedValue(device);
            repository.update.mockResolvedValue(updatedDevice);

            const result = await service.touchKnownDevice(
                'device-id',
                { lastIpAddress: '1.2.3.4' },
                120,
                undefined,
                now,
            );

            expect(result).toEqual(updatedDevice);
            expect(repository.update).toHaveBeenCalledWith(
                'device-id',
                {
                    lastSeenAt: now,
                    lastIpAddress: '1.2.3.4',
                },
                undefined,
            );
        });
    });

    describe('trustKnownDevice', () => {
        it('should return null if device not found', async () => {
            repository.findById.mockResolvedValue(null);

            const result = await service.trustKnownDevice('device-id');

            expect(result).toBeNull();
            expect(repository.update).not.toHaveBeenCalled();
        });

        it('should update trust fields and return trusted device', async () => {
            const now = new Date();
            const future = new Date(now.getTime() + 100000);
            const device = buildMockDevice({ trustedAt: null });
            const trustedDevice = buildMockDevice({
                trustedAt: now,
                trustExpiresAt: future,
            });
            repository.findById.mockResolvedValue(device);
            repository.update.mockResolvedValue(trustedDevice);

            const result = await service.trustKnownDevice('device-id', future, undefined, now);

            expect(result).toEqual(trustedDevice);
            expect(repository.update).toHaveBeenCalledWith(
                'device-id',
                {
                    trustedAt: now,
                    trustExpiresAt: future,
                },
                undefined,
            );
        });
    });

    describe('revokeKnownDevice', () => {
        it('should throw NotFoundException if device not found', async () => {
            repository.findByIdForUpdate.mockResolvedValue(null);

            await expect(
                service.revokeKnownDevice('user-uuid', 'customer', 'device-id'),
            ).rejects.toThrow(NotFoundException);
        });

        it('should throw ForbiddenException if ownership check fails for userId', async () => {
            const device = buildMockDevice({ userId: 'other-user', realm: 'customer' });
            repository.findByIdForUpdate.mockResolvedValue(device);

            await expect(
                service.revokeKnownDevice('user-uuid', 'customer', 'device-id'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should throw ForbiddenException if ownership check fails for realm', async () => {
            const device = buildMockDevice({ userId: 'user-uuid', realm: 'admin' });
            repository.findByIdForUpdate.mockResolvedValue(device);

            await expect(
                service.revokeKnownDevice('user-uuid', 'customer', 'device-id'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should return and do nothing if device is already revoked', async () => {
            const device = buildMockDevice({
                userId: 'user-uuid',
                realm: 'customer',
                revokedAt: new Date(),
            });
            repository.findByIdForUpdate.mockResolvedValue(device);

            await service.revokeKnownDevice('user-uuid', 'customer', 'device-id');

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(repository.revoke).not.toHaveBeenCalled();
            expect(sessionRepository.revokeUnrevokedByKnownDevice).not.toHaveBeenCalled();
        });

        it('should successfully revoke device, its active sessions, blacklist them, and save security events', async () => {
            const device = buildMockDevice({ userId: 'user-uuid', realm: 'customer' });
            const session = buildMockSession({
                id: 'session-id',
                userId: 'user-uuid',
                realm: 'customer',
                ipAddress: '1.2.3.4',
                userAgent: 'chrome',
            });

            repository.findByIdForUpdate.mockResolvedValue(device);
            sessionRepository.findUnrevokedByKnownDeviceForUpdate.mockResolvedValue([session]);

            await service.revokeKnownDevice('user-uuid', 'customer', 'device-id');

            // Blacklist (15m * 60 = 900s + 60s clock skew = 960s)
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('session-id', 960);
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledWith(
                'session-id',
                expect.any(Date),
                expect.any(Object),
            );

            expect(repository.revoke).toHaveBeenCalledWith(
                'device-id',
                expect.any(Date),
                expect.any(Object),
            );
            expect(sessionRepository.revokeUnrevokedByKnownDevice).toHaveBeenCalledWith(
                'device-id',
                expect.any(Date),
                expect.any(Object),
            );

            // Security Events
            expect(securityEventRepository.save).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    sessionId: 'session-id',
                    eventType: SecurityEventType.SESSION_REVOKED,
                    ipAddress: '1.2.3.4',
                    userAgent: 'chrome',
                    metadata: {
                        revocationReason: 'known_device_revoked',
                    },
                },
                expect.any(Object),
            );
        });

        it('should fail-closed by rolling back DB write if Redis blacklist fails', async () => {
            const device = buildMockDevice({ userId: 'user-uuid', realm: 'customer' });
            const session = buildMockSession({ id: 'session-id' });

            repository.findByIdForUpdate.mockResolvedValue(device);
            sessionRepository.findUnrevokedByKnownDeviceForUpdate.mockResolvedValue([session]);

            redisTokenService.revokeSession.mockRejectedValue(new Error('Redis write failed'));

            await expect(
                service.revokeKnownDevice('user-uuid', 'customer', 'device-id'),
            ).rejects.toThrow('Redis write failed');

            expect(repository.revoke).not.toHaveBeenCalled();
            expect(sessionRepository.revokeUnrevokedByKnownDevice).not.toHaveBeenCalled();
        });
    });
});
