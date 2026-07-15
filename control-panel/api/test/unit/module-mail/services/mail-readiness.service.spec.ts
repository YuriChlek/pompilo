import { Test, TestingModule } from '@nestjs/testing';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';

describe('MailReadinessService', () => {
    let service: MailReadinessService;
    let repository: { findSingleton: jest.Mock };
    let settingsService: { getCachedSettings: jest.Mock };
    let redisService: {
        getFailureCount: jest.Mock;
        getDeliveryErrors: jest.Mock;
        getLastSuccessfulSendAt: jest.Mock;
        getReadinessStatus: jest.Mock;
        setReadinessStatus: jest.Mock;
    };
    let encryptionService: {
        decryptMailSecret: jest.Mock;
        isKeyAvailable: jest.Mock;
    };
    let mailService: {
        verifyTransport: jest.Mock;
    };

    beforeEach(async () => {
        const findSingletonMock = jest.fn();
        repository = { findSingleton: findSingletonMock };
        settingsService = {
            getCachedSettings: jest
                .fn()
                .mockImplementation(
                    (tx?: unknown) => repository.findSingleton(tx) as Promise<unknown>,
                ),
        };

        /* eslint-disable @typescript-eslint/unbound-method, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return */
        const originalMockResolvedValue = findSingletonMock.mockResolvedValue;
        findSingletonMock.mockResolvedValue = (value: any) => {
            if (value && typeof value === 'object' && !('clientPublicUrl' in value)) {
                value.clientPublicUrl = 'http://localhost:3000';
            }
            return originalMockResolvedValue.call(findSingletonMock, value);
        };
        /* eslint-enable @typescript-eslint/unbound-method, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return */
        redisService = {
            getFailureCount: jest.fn().mockResolvedValue(0),
            getDeliveryErrors: jest.fn().mockResolvedValue([]),
            getLastSuccessfulSendAt: jest.fn().mockResolvedValue(null),
            getReadinessStatus: jest.fn().mockResolvedValue(null),
            setReadinessStatus: jest.fn(),
        };
        encryptionService = {
            decryptMailSecret: jest.fn().mockReturnValue('decrypted'),
            isKeyAvailable: jest.fn().mockReturnValue(true),
        };
        mailService = {
            verifyTransport: jest.fn(),
        };
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailReadinessService,
                {
                    provide: MailSettingsRepository,
                    useValue: repository as unknown as MailSettingsRepository,
                },
                {
                    provide: MailRedisService,
                    useValue: redisService as unknown as MailRedisService,
                },
                {
                    provide: MailEncryptionService,
                    useValue: encryptionService as unknown as MailEncryptionService,
                },
                {
                    provide: MAIL_SERVICE,
                    useValue: mailService,
                },
                {
                    provide: MailSettingsService,
                    useValue: settingsService as unknown as MailSettingsService,
                },
            ],
        }).compile();

        service = module.get<MailReadinessService>(MailReadinessService);
    });

    describe('getHealth', () => {
        it('should respect Redis-backed readiness status if present', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            redisService.getReadinessStatus.mockResolvedValue({
                status: 'unhealthy',
                error: 'circuit_breaker_active',
                timestamp: new Date().toISOString(),
            });

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('circuit_breaker_active');
            expect(health.isDegraded).toBe(false);
        });

        it('should fallback to local memory if Redis is unavailable', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            redisService.getReadinessStatus.mockRejectedValue(new Error('Redis down'));
            redisService.setReadinessStatus.mockRejectedValue(new Error('Redis down'));

            // First set status (it will fail Redis and use local fallback)
            await service.setReadinessStatus('unhealthy', 'local_error');

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('local_error');
            expect(health.isDegraded).toBe(true);
        });

        it('should clear local fallback after TTL', async () => {
            jest.useFakeTimers();
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            redisService.getReadinessStatus.mockRejectedValue(new Error('Redis down'));
            redisService.setReadinessStatus.mockRejectedValue(new Error('Redis down'));

            await service.setReadinessStatus('unhealthy', 'local_error', 1); // 1 second TTL

            let health = await service.getHealth();
            expect(health.status).toBe('unhealthy');

            jest.advanceTimersByTime(1100);

            health = await service.getHealth();
            expect(health.status).toBe('healthy');
            jest.useRealTimers();
        });

        it('should return unhealthy if no settings exist', async () => {
            repository.findSingleton.mockResolvedValue(null);
            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('mail_settings_missing');
        });

        it('should return disabled if settings are disabled', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: false,
            });
            const health = await service.getHealth();
            expect(health.status).toBe('disabled');
        });

        it('should return healthy without prior manual verification', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: null,
            });
            const health = await service.getHealth();
            expect(health.status).toBe('healthy');
        });

        it('should return unhealthy if encryption key is missing', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            encryptionService.isKeyAvailable.mockReturnValue(false);

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('encryption_key_missing');
        });

        it('should return unhealthy if encryption key is mismatched', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
                smtpPasswordEncrypted: 'enc:secret',
            });
            encryptionService.decryptMailSecret.mockImplementation(() => {
                throw new Error('encryption_key_mismatch');
            });

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('encryption_key_mismatch');
        });

        it('should return healthy if settings are enabled and verified', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            const health = await service.getHealth();
            expect(health.status).toBe('healthy');
        });

        it('should return unhealthy if too many failures in Redis', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
            });
            redisService.getFailureCount.mockResolvedValue(11);

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('too_many_delivery_failures');
        });

        it('should return unhealthy if clientPublicUrl is missing', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                lastVerifiedAt: new Date(),
                clientPublicUrl: null,
            });

            const health = await service.getHealth();
            expect(health.status).toBe('unhealthy');
            expect(health.lastError).toBe('mail_settings_client_url_missing');
        });
    });

    describe('assertMailReadyForCriticalFlow', () => {
        it('should throw if settings are missing (unconfigured)', async () => {
            repository.findSingleton.mockResolvedValue(null);
            await expect(service.assertMailReadyForCriticalFlow()).rejects.toThrow(
                'mail_service_disabled_or_unconfigured',
            );
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
        });

        it('should throw if disabled', async () => {
            repository.findSingleton.mockResolvedValue({ enabled: false });
            await expect(service.assertMailReadyForCriticalFlow()).rejects.toThrow(
                'mail_service_disabled_or_unconfigured',
            );
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
        });

        it('should succeed without confirmation in any environment', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                confirmedAt: null,
                lastVerifiedAt: new Date(),
            });
            await expect(service.assertMailReadyForCriticalFlow()).resolves.not.toThrow();
        });

        it('should throw if technical health is unhealthy', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                confirmedAt: new Date(),
                lastVerifiedAt: new Date(),
            });
            // Mock getHealth failure via too many errors
            redisService.getFailureCount.mockResolvedValue(20);

            await expect(service.assertMailReadyForCriticalFlow()).rejects.toThrow(
                'mail_service_not_ready: too_many_delivery_failures',
            );
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
        });

        it('should succeed if fully configured and healthy', async () => {
            repository.findSingleton.mockResolvedValue({
                enabled: true,
                confirmedAt: new Date(),
                lastVerifiedAt: new Date(),
            });
            await expect(service.assertMailReadyForCriticalFlow()).resolves.not.toThrow();
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
        });
    });
});
