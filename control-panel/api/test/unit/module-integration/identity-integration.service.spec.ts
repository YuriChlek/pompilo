import { ForbiddenException, ServiceUnavailableException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';
import { UserRepository } from '@/module-user/repository/user.repository';
import { IdentityOutboxRepository } from '@/module-integration/repository/identity-outbox.repository';
import { ConfigService } from '@nestjs/config';
import { UserRoles } from '@/module-auth/enums/auth.enums';

describe('IdentityIntegrationService', () => {
    let service: IdentityIntegrationService;
    let userRepository: jest.Mocked<UserRepository>;
    let outboxRepository: jest.Mocked<IdentityOutboxRepository>;
    let configService: jest.Mocked<ConfigService>;
    let jwtService: jest.Mocked<JwtService>;

    beforeEach(() => {
        userRepository = {
            findPrimaryMembership: jest.fn().mockResolvedValue({
                userId: 'user-1',
                tenantId: 'tenant-1',
                role: 'OWNER',
            }),
            findById: jest.fn().mockResolvedValue({
                id: 'user-1',
                email: 'user@example.com',
                name: 'User',
                role: UserRoles.USER,
                emailVerifiedAt: new Date(),
            }),
        } as unknown as jest.Mocked<UserRepository>;
        outboxRepository = {
            create: jest.fn().mockResolvedValue({ id: 'outbox-1' }),
        } as unknown as jest.Mocked<IdentityOutboxRepository>;
        configService = {
            get: jest.fn((key: string) => {
                if (key === 'TRADING_ONBOARDING_TOKEN_SECRET') {
                    return 'x'.repeat(32);
                }
                if (key === 'TRADING_ONBOARDING_TOKEN_TTL_SECONDS') {
                    return 120;
                }
                return undefined;
            }),
        } as unknown as jest.Mocked<ConfigService>;
        jwtService = {
            signAsync: jest.fn().mockResolvedValue('signed-token'),
        } as unknown as jest.Mocked<JwtService>;

        service = new IdentityIntegrationService(
            userRepository,
            outboxRepository,
            configService,
            jwtService,
        );
    });

    it('records UserRegistered into the transactional outbox', async () => {
        await service.recordUserRegistered(
            {
                id: 'user-1',
                email: 'user@example.com',
                name: 'User',
                role: UserRoles.USER,
            },
            'tenant-1',
            {} as never,
        );

        expect(outboxRepository.create).toHaveBeenCalledWith(
            expect.objectContaining({
                eventType: 'identity.v1.UserRegistered',
                aggregateId: 'user-1',
                tenantId: 'tenant-1',
                eventVersion: 1,
                idempotencyKey: 'identity.v1.UserRegistered:user-1:tenant-1:1',
            }),
            expect.any(Object),
        );
    });

    it('creates short-lived trading onboarding tokens for verified users', async () => {
        const result = await service.createTradingOnboardingToken('user-1');

        expect(result).toEqual({
            token: 'signed-token',
            expiresInSeconds: 120,
            tokenType: 'Bearer',
        });
        expect(jwtService.signAsync).toHaveBeenCalledWith(
            expect.objectContaining({
                aud: 'trading-service',
                scope: 'trading:onboarding',
                userId: 'user-1',
                tenantId: 'tenant-1',
                emailVerified: true,
            }),
            expect.objectContaining({
                secret: 'x'.repeat(32),
                expiresIn: 120,
            }),
        );
    });

    it('rejects onboarding token creation before email verification', async () => {
        userRepository.findById.mockResolvedValueOnce({
            id: 'user-1',
            role: UserRoles.USER,
            emailVerifiedAt: null,
        } as never);

        await expect(service.createTradingOnboardingToken('user-1')).rejects.toThrow(
            ForbiddenException,
        );
    });

    it('requires a membership context', async () => {
        userRepository.findPrimaryMembership.mockResolvedValueOnce(null);

        await expect(service.getIdentityContext('user-1')).rejects.toThrow(
            ServiceUnavailableException,
        );
    });
});
