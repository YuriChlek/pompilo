/* eslint-disable @typescript-eslint/unbound-method */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
import {
    BadRequestException,
    InternalServerErrorException,
    UnauthorizedException,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import { CustomerAuthService } from '@/module-customer-auth/services/customer-auth.service';
import { AuthService } from '@/module-auth/services/auth.service';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import { RegisterDto } from '@/module-auth/dto/register-user.dto';
import { LoginUserDto } from '@/module-auth/dto/login-user.dto';
import { UserRepository } from '@/module-user/repository/user.repository';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { UserUniquenessService } from '@/module-user/services/user-uniqueness.service';
import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { EmailVerificationService } from '@/module-auth-token/services/email-verification.service';
import { SessionService } from '@/module-auth-token/services/session.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import {
    TransactionRepository,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

type AwaitedReturn<T> = T extends Promise<infer R> ? R : T;

describe('CustomerAuthService', () => {
    let service: CustomerAuthService;
    let authService: {
        login: jest.MockedFunction<AuthService['login']>;
        logout: jest.MockedFunction<AuthService['logout']>;
        refreshAccessToken: jest.MockedFunction<AuthService['refreshAccessToken']>;
        getMe: jest.MockedFunction<AuthService['getMe']>;
    };
    let userRepository: jest.Mocked<UserRepository>;
    let userPasswordService: jest.Mocked<UserPasswordService>;
    let userUniquenessService: jest.Mocked<UserUniquenessService>;
    let deviceIdService: jest.Mocked<DeviceIdService>;
    let knownDeviceService: jest.Mocked<KnownDeviceService>;
    let sessionService: jest.Mocked<SessionService>;
    let securityEventService: jest.Mocked<SecurityEventService>;
    let geoIpService: jest.Mocked<GeoIpService>;
    let transactionRepository: {
        run: jest.Mock;
    };
    let authSessionService: jest.Mocked<AuthSessionService>;
    let emailVerificationService: jest.Mocked<EmailVerificationService>;
    let identityIntegrationService: jest.Mocked<IdentityIntegrationService>;

    const request = {} as Request;
    const response = {} as Response;

    beforeEach(() => {
        request.cookies = {};
        authService = {
            login: jest
                .fn()
                .mockResolvedValue({ id: '1' } as AwaitedReturn<ReturnType<AuthService['login']>>),
            logout: jest.fn().mockResolvedValue(true),
            refreshAccessToken: jest.fn().mockResolvedValue(true),
            getMe: jest.fn().mockReturnValue({ id: '1' }),
        };

        userRepository = {
            createWithTenant: jest.fn().mockResolvedValue({
                user: {
                    id: '3',
                    name: 'User',
                    email: 'user@example.com',
                    role: UserRoles.USER,
                },
                tenant: { id: 'tenant-id', name: "User's Space" },
                membership: { id: 'membership-id' },
            }),
            findById: jest.fn(),
        } as unknown as jest.Mocked<UserRepository>;

        userPasswordService = {
            hashPassword: jest.fn().mockResolvedValue('hashed_password'),
            comparePassword: jest.fn(),
        } as unknown as jest.Mocked<UserPasswordService>;

        userUniquenessService = {
            ensureUnique: jest.fn().mockResolvedValue(undefined),
        } as unknown as jest.Mocked<UserUniquenessService>;

        deviceIdService = {
            getOrCreateDeviceId: jest.fn().mockReturnValue({ deviceId: 'device-uuid' }),
            setDeviceIdCookie: jest.fn(),
        } as unknown as jest.Mocked<DeviceIdService>;

        knownDeviceService = {
            findOrCreateKnownDevice: jest.fn().mockResolvedValue({ id: 'known-device-id' }),
        } as unknown as jest.Mocked<KnownDeviceService>;

        sessionService = {
            createSession: jest.fn().mockResolvedValue({ id: 'session-uuid' }),
        } as unknown as jest.Mocked<SessionService>;

        securityEventService = {
            recordRegistrationSuccess: jest.fn().mockResolvedValue(null),
        } as unknown as jest.Mocked<SecurityEventService>;

        geoIpService = {
            lookup: jest.fn().mockResolvedValue({ country: 'US', region: 'CA', city: 'SF' }),
        } as unknown as jest.Mocked<GeoIpService>;

        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation(
                    async (work: (tx: RepositoryTransaction) => Promise<unknown>) => {
                        const mockTx = {} as RepositoryTransaction;
                        return await work(mockTx);
                    },
                ),
        };

        authSessionService = {
            getUserMetaData: jest
                .fn()
                .mockReturnValue({ ipAddress: '127.0.0.1', userAgent: 'jest' }),
            issueTokensDeferred: jest.fn().mockResolvedValue({
                accessToken: 'access-token',
                refreshToken: 'refresh-token',
            }),
            applyTokens: jest.fn(),
        } as unknown as jest.Mocked<AuthSessionService>;

        emailVerificationService = {
            createVerificationToken: jest.fn().mockResolvedValue('verification-token'),
            sendVerificationEmail: jest.fn().mockResolvedValue(undefined),
            verifyEmail: jest.fn().mockResolvedValue(true),
            resendVerification: jest.fn().mockResolvedValue(undefined),
        } as unknown as jest.Mocked<EmailVerificationService>;

        identityIntegrationService = {
            recordUserRegistered: jest.fn().mockResolvedValue(undefined),
            recordEmailVerified: jest.fn().mockResolvedValue(undefined),
        } as unknown as jest.Mocked<IdentityIntegrationService>;

        service = new CustomerAuthService(
            authService as unknown as AuthService,
            userRepository,
            userPasswordService,
            userUniquenessService,
            deviceIdService,
            knownDeviceService,
            sessionService,
            securityEventService,
            geoIpService,
            transactionRepository as unknown as TransactionRepository,
            authSessionService,
            emailVerificationService,
            identityIntegrationService,
        );
    });

    it('registers generic users through UserRepository and issues tokens', async () => {
        const dto: RegisterDto = {
            email: 'user@example.com',
            password: 'Secret123',
            name: 'User',
        };

        const result = await service.register(response, request, dto);

        expect(userUniquenessService.ensureUnique).toHaveBeenCalledWith(
            dto.email,
            dto.name,
            undefined,
            expect.any(Object),
        );
        expect(userPasswordService.hashPassword).toHaveBeenCalledWith(dto.password);
        expect(userRepository.createWithTenant).toHaveBeenCalledWith(
            {
                name: dto.name,
                email: dto.email,
                password: 'hashed_password',
                role: UserRoles.USER,
            },
            "User's Space",
            expect.any(Object),
        );
        expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(response, 'device-uuid');
        expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
            accessToken: 'access-token',
            refreshToken: 'refresh-token',
            role: UserRoles.USER,
        });
        expect(securityEventService.recordRegistrationSuccess).toHaveBeenCalled();
        expect(emailVerificationService.createVerificationToken).toHaveBeenCalledWith(
            '3',
            expect.any(Object),
        );
        expect(emailVerificationService.sendVerificationEmail).toHaveBeenCalledWith(
            '3',
            'user@example.com',
            'User',
            'verification-token',
            expect.any(Object),
        );
        expect(identityIntegrationService.recordUserRegistered).toHaveBeenCalledWith(
            expect.objectContaining({ id: '3', email: 'user@example.com', role: UserRoles.USER }),
            'tenant-id',
            expect.any(Object),
        );
        expect(result).toEqual({
            id: '3',
            name: 'User',
            email: 'user@example.com',
            role: UserRoles.USER,
        });
    });

    it('logs user in via AuthService using the customer role', async () => {
        const dto: LoginUserDto = {
            login: 'john',
            password: 'Secret123',
        };

        await service.login(response, request, dto);

        expect(authService.login).toHaveBeenCalledWith(response, request, dto, [UserRoles.USER]);
    });

    it('rejects admin roles on customer auth entry points', async () => {
        authService.login.mockRejectedValue(new UnauthorizedException());

        await expect(
            service.login(response, request, {
                login: 'admin@example.com',
                password: 'Secret123',
            }),
        ).rejects.toBeInstanceOf(UnauthorizedException);
    });

    it('logs user out through AuthService', async () => {
        await service.logout(request, response, UserRoles.USER);
        expect(authService.logout).toHaveBeenCalledWith(request, response, UserRoles.USER);

        await service.logout(request, response, UserRoles.USER);
        expect(authService.logout).toHaveBeenCalledWith(request, response, UserRoles.USER);
    });

    it('propagates logout failures', async () => {
        const error = new Error('redis unavailable');
        authService.logout.mockRejectedValue(error);

        await expect(service.logout(request, response, UserRoles.USER)).rejects.toBe(error);
    });

    it('refreshes customer token using customerRefreshToken cookie', async () => {
        request.cookies = {
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'customer-refresh-token',
        };

        const result = await service.refresh(response, request, UserRoles.USER);

        expect(result).toBe(true);
        expect(authService.refreshAccessToken).toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
        );
    });

    it('customer refresh only reads customerRefreshToken and ignores adminRefreshToken', async () => {
        request.cookies = {
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'customer-refresh-token',
            [COOKIE_NAMES.ADMIN_REFRESH_TOKEN]: 'admin-refresh-token',
        };

        const result = await service.refresh(response, request, UserRoles.USER);

        expect(result).toBe(true);
        expect(authService.refreshAccessToken).toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
        );
        expect(authService.refreshAccessToken).not.toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
        );
    });

    it('throws UnauthorizedException during refresh if customerRefreshToken cookie is missing', async () => {
        request.cookies = {};

        await expect(service.refresh(response, request, UserRoles.USER)).rejects.toBeInstanceOf(
            UnauthorizedException,
        );
        expect(authService.refreshAccessToken).not.toHaveBeenCalled();
    });

    it('wraps refresh errors into InternalServerErrorException', async () => {
        request.cookies = {
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'customer-refresh-token',
        };
        authService.refreshAccessToken.mockRejectedValue(new Error('failed'));

        await expect(service.refresh(response, request, UserRoles.USER)).rejects.toBeInstanceOf(
            InternalServerErrorException,
        );
    });

    it('rethrows UnauthorizedException coming from AuthService', async () => {
        request.cookies = {
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'customer-refresh-token',
        };
        const unauthorized = new UnauthorizedException();
        authService.refreshAccessToken.mockRejectedValue(unauthorized);

        await expect(service.refresh(response, request, UserRoles.USER)).rejects.toBe(
            unauthorized,
        );
    });

    it('delegates getMe lookup to AuthService', () => {
        const result = service.getMe(request, UserRoles.USER);

        expect(result).toEqual({ id: '1' });
        expect(authService.getMe).toHaveBeenCalledWith(request, UserRoles.USER);
    });

    describe('verifyEmail', () => {
        it('should delegate verifyEmail to EmailVerificationService', async () => {
            emailVerificationService.verifyEmail.mockResolvedValue(true);
            const result = await service.verifyEmail('token-123');
            expect(result).toBe(true);
            expect(emailVerificationService.verifyEmail).toHaveBeenCalledWith(
                'token-123',
                undefined,
                expect.any(Function),
            );
        });

        it('passes a transactional integration callback to email verification', async () => {
            emailVerificationService.verifyEmail.mockImplementation(
                async (_token, _transaction, onVerified) => {
                    await onVerified?.(
                        {
                            id: 'user-123',
                            name: 'User',
                            email: 'user@example.com',
                            role: UserRoles.USER,
                        } as any,
                        {} as RepositoryTransaction,
                    );
                    return true;
                },
            );

            await service.verifyEmail('token-123');

            expect(identityIntegrationService.recordEmailVerified).toHaveBeenCalledWith(
                expect.objectContaining({ id: 'user-123', role: UserRoles.USER }),
                expect.any(Object),
            );
        });
    });

    describe('resendVerification', () => {
        it('should throw BadRequestException if user is not found', async () => {
            userRepository.findById.mockResolvedValue(null);
            await expect(service.resendVerification('user-123')).rejects.toThrow(
                BadRequestException,
            );
        });

        it('should delegate resendVerification to EmailVerificationService', async () => {
            userRepository.findById.mockResolvedValue({
                id: 'user-123',
                email: 'user@example.com',
                name: 'User',
            } as any);
            await service.resendVerification('user-123');
            expect(emailVerificationService.resendVerification).toHaveBeenCalledWith(
                'user-123',
                'user@example.com',
                'User',
            );
        });
    });
});
