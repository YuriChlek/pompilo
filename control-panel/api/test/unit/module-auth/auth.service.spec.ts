import {
    ConflictException,
    HttpException,
    HttpStatus,
    InternalServerErrorException,
    ServiceUnavailableException,
    UnauthorizedException,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import { AuthService } from '@/module-auth/services/auth.service';
import { UserService } from '@/module-user/services/user.service';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { SessionService } from '@/module-auth-token/services/session.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import { TransactionRepository } from '@/module-drizzle/repository/transaction.repository';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { buildRegisterDto, buildLoginDto } from '../../fixtures/auth.fixtures';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { ConfigService } from '@nestjs/config';
import { RiskPolicyService } from '@/module-auth-token/services/risk-policy.service';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { LoginChallengeResendPolicyService } from '@/module-auth-token/services/login-challenge-resend-policy.service';

type AwaitedReturn<T> = T extends Promise<infer R> ? R : T;
const hashMock = jest.spyOn(Argon2HashUtil, 'hash');
const compareMock = jest.spyOn(Argon2HashUtil, 'compare');

const createRequest = (overrides?: Partial<Request>): Request =>
    ({
        cookies: {},
        headers: {
            'x-forwarded-for': '203.0.113.1',
            'user-agent': 'Jest',
        },
        socket: {
            remoteAddress: '10.0.0.1',
        },
        ...overrides,
    }) as Request;

const createResponse = (): Response =>
    ({
        cookie: jest.fn(),
    }) as unknown as Response;

describe('AuthService', () => {
    let service: AuthService;
    let userService: {
        create: jest.MockedFunction<UserService['create']>;
        findByLogin: jest.MockedFunction<UserService['findByLogin']>;
        findById: jest.MockedFunction<UserService['findById']>;
    };
    let tokenService: {
        verifyRefreshToken: jest.Mock;
        removeRefreshToken: jest.Mock;
        revokeAccessSession: jest.Mock;
        getTokenData: jest.Mock;
        rotateRefreshTokenById: jest.Mock;
        createAccessToken: jest.Mock;
        revokeTokensBySession: jest.Mock;
        acceptReplacedTokenInGraceById: jest.Mock;
    };
    let authSessionService: {
        issueTokens: jest.Mock;
        clearTokens: jest.Mock;
        getCookieName: jest.Mock;
        getUserMetaData: jest.Mock;
        clearOppositeCustomerTokens: jest.Mock;
        applyTokens: jest.Mock;
        issueTokensDeferred: jest.Mock;
    };
    let deviceIdService: {
        getOrCreateDeviceId: jest.Mock;
        setDeviceIdCookie: jest.Mock;
        readDeviceId: jest.Mock;
    };
    let knownDeviceService: {
        findOrCreateKnownDevice: jest.Mock;
        listKnownDevices: jest.Mock;
        touchKnownDevice: jest.Mock;
        trustKnownDevice: jest.Mock;
    };
    let sessionService: {
        findReusableSession: jest.Mock;
        createSession: jest.Mock;
        reuseSession: jest.Mock;
        getSession: jest.Mock;
        touchLastSeenAt: jest.Mock;
        revokeCurrentSession: jest.Mock;
    };
    let securityEventService: {
        recordLoginSuccess: jest.Mock;
        recordRefreshFailed: jest.Mock;
        recordSuspiciousDevice: jest.Mock;
        recordLoginApprovalRequired: jest.Mock;
        recordLoginFailed: jest.Mock;
        recordLoginApprovalPassed: jest.Mock;
        recordLoginApprovalFailed: jest.Mock;
        recordLoginApprovalResent: jest.Mock;
        recordLoginApprovalResendFailed: jest.Mock;
    };
    let geoIpService: {
        lookup: jest.Mock;
    };
    let transactionRepository: {
        run: jest.Mock;
    };
    let configService: {
        get: jest.Mock;
    };
    let riskPolicyService: {
        evaluateRisk: jest.Mock;
    };
    let securityEventRepository: {
        findByUserId: jest.Mock;
    };
    let loginChallengeService: {
        createLoginChallenge: jest.Mock;
        getChallenge: jest.Mock;
        getLatestChallengeForDevice: jest.Mock;
        getResendEligibility: jest.Mock;
        verifyLoginChallenge: jest.Mock;
        approveAndConsumeLoginChallengeAtomically: jest.Mock;
    };
    let mailTemplateService: {
        sendVerificationCode: jest.MockedFunction<MailTemplateService['sendVerificationCode']>;
        sendVerificationCodeOrThrow: jest.MockedFunction<
            MailTemplateService['sendVerificationCodeOrThrow']
        >;
        sendSecurityAlert: jest.MockedFunction<MailTemplateService['sendSecurityAlert']>;
    };
    let loginChallengeResendPolicyService: {
        reserveResendAttempt: jest.MockedFunction<
            LoginChallengeResendPolicyService['reserveResendAttempt']
        >;
    };
    let response: Response;
    let request: Request;

    const registerDto = buildRegisterDto();
    const loginDto = buildLoginDto({ login: registerDto.email, role: UserRoles.USER });

    beforeEach(() => {
        userService = {
            create: jest.fn(),
            findByLogin: jest.fn(),
            findById: jest.fn(),
        };

        tokenService = {
            verifyRefreshToken: jest.fn().mockResolvedValue({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                user: { id: 'user-id' },
                isGrace: false,
            }),
            removeRefreshToken: jest.fn(),
            revokeAccessSession: jest.fn(),
            getTokenData: jest.fn().mockReturnValue({ userId: 'user-id', sessionId: 'session-id' }),
            revokeTokensBySession: jest.fn(),
            rotateRefreshTokenById: jest.fn().mockResolvedValue('new-refresh-token'),
            createAccessToken: jest.fn().mockReturnValue('new-access-token'),
            acceptReplacedTokenInGraceById: jest
                .fn()
                .mockResolvedValue('replacement-refresh-token'),
        };

        authSessionService = {
            issueTokens: jest.fn(),
            clearTokens: jest.fn(),
            getCookieName: jest.fn((userRole: UserRoles, tokenType: TokenType) => {
                if (userRole === UserRoles.PLATFORM_ADMIN || userRole === UserRoles.SUPER_ADMIN) {
                    return tokenType === TokenType.ACCESS
                        ? COOKIE_NAMES.ADMIN_ACCESS_TOKEN
                        : COOKIE_NAMES.ADMIN_REFRESH_TOKEN;
                }

                return tokenType === TokenType.ACCESS
                    ? COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN
                    : COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN;
            }),
            getUserMetaData: jest.fn().mockReturnValue({
                ipAddress: '203.0.113.1',
                userAgent: 'Jest',
            }),
            clearOppositeCustomerTokens: jest.fn(),
            applyTokens: jest.fn(),
            issueTokensDeferred: jest.fn().mockResolvedValue({
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                payloadWithSession: {
                    role: UserRoles.USER,
                },
            }),
        };

        deviceIdService = {
            getOrCreateDeviceId: jest.fn().mockReturnValue({ deviceId: 'device-id', isNew: false }),
            setDeviceIdCookie: jest.fn(),
            readDeviceId: jest.fn().mockReturnValue('device-id'),
        };

        knownDeviceService = {
            findOrCreateKnownDevice: jest.fn().mockResolvedValue({ id: 'test-known-device-id' }),
            listKnownDevices: jest.fn().mockResolvedValue([]),
            touchKnownDevice: jest.fn().mockResolvedValue(null),
            trustKnownDevice: jest.fn().mockResolvedValue({}),
        };

        sessionService = {
            findReusableSession: jest.fn().mockResolvedValue(null),
            createSession: jest.fn().mockResolvedValue({ id: 'test-session-id' }),
            reuseSession: jest.fn().mockResolvedValue({ id: 'test-session-id' }),
            getSession: jest.fn().mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
            }),
            touchLastSeenAt: jest.fn().mockResolvedValue(null),
            revokeCurrentSession: jest.fn().mockResolvedValue(undefined),
        };

        securityEventService = {
            recordLoginSuccess: jest.fn().mockResolvedValue(null),
            recordRefreshFailed: jest.fn().mockResolvedValue(null),
            recordSuspiciousDevice: jest.fn().mockResolvedValue(null),
            recordLoginApprovalRequired: jest.fn().mockResolvedValue(null),
            recordLoginFailed: jest.fn().mockResolvedValue(null),
            recordLoginApprovalPassed: jest.fn().mockResolvedValue(null),
            recordLoginApprovalFailed: jest.fn().mockResolvedValue(null),
            recordLoginApprovalResent: jest.fn().mockResolvedValue(null),
            recordLoginApprovalResendFailed: jest.fn().mockResolvedValue(null),
        };

        geoIpService = {
            lookup: jest.fn().mockResolvedValue({ country: 'US', region: 'CA', city: 'San Jose' }),
        };

        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation(<T>(work: (tx: any) => Promise<T>) =>
                    work('test-transaction' as any),
                ),
        };

        configService = {
            get: jest.fn().mockReturnValue(false),
        };

        riskPolicyService = {
            evaluateRisk: jest.fn().mockReturnValue({
                score: 0,
                decision: 'low',
                reasons: [],
                requiredAction: 'allow',
            }),
        };

        securityEventRepository = {
            findByUserId: jest.fn().mockResolvedValue([]),
        };

        loginChallengeService = {
            createLoginChallenge: jest.fn().mockResolvedValue({
                challenge: { id: 'mock-challenge-uuid' },
                checkpointToken: 'mock-checkpoint-token',
                code: '123456',
            }),
            getChallenge: jest.fn(),
            getLatestChallengeForDevice: jest.fn(),
            getResendEligibility: jest.fn(),
            verifyLoginChallenge: jest.fn(),
            approveAndConsumeLoginChallengeAtomically: jest.fn(),
        };

        mailTemplateService = {
            sendVerificationCode: jest.fn().mockResolvedValue({}),
            sendVerificationCodeOrThrow: jest.fn().mockResolvedValue({}),
            sendSecurityAlert: jest.fn().mockResolvedValue({}),
        };

        loginChallengeResendPolicyService = {
            reserveResendAttempt: jest.fn().mockResolvedValue({
                allowed: true,
                retryAfterSeconds: 0,
            }),
        };

        service = new AuthService(
            userService as unknown as UserService,
            tokenService as unknown as AuthTokenService,
            authSessionService as unknown as AuthSessionService,
            deviceIdService as unknown as DeviceIdService,
            knownDeviceService as unknown as KnownDeviceService,
            sessionService as unknown as SessionService,
            securityEventService as unknown as SecurityEventService,
            geoIpService as unknown as GeoIpService,
            transactionRepository as unknown as TransactionRepository,
            configService as unknown as ConfigService,
            riskPolicyService as unknown as RiskPolicyService,
            securityEventRepository as unknown as SecurityEventRepository,
            loginChallengeService as unknown as LoginChallengeService,
            mailTemplateService as unknown as MailTemplateService,
            loginChallengeResendPolicyService as unknown as LoginChallengeResendPolicyService,
        );

        response = createResponse();
        request = createRequest();
        jest.clearAllMocks();
        hashMock.mockResolvedValue('hashed');
        compareMock.mockReset();
    });

    describe('register', () => {
        it('registers a new user and sets cookies', async () => {
            userService.create.mockResolvedValue({
                id: 'user-id',
                name: 'John Doe',
                email: 'john.doe@example.com',
                role: UserRoles.USER,
            } as AwaitedReturn<ReturnType<UserService['create']>>);

            const result = await service.register(response, request, registerDto);

            expect(userService.create).toHaveBeenCalledWith(registerDto);
            expect(deviceIdService.getOrCreateDeviceId).toHaveBeenCalledWith(request);
            expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(response, 'device-id');
            expect(authSessionService.issueTokens).toHaveBeenCalledWith(
                response,
                expect.objectContaining({
                    id: 'user-id',
                    ipAddress: '203.0.113.1',
                    userAgent: 'Jest',
                }),
            );
            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: 'john.doe@example.com',
                role: UserRoles.USER,
            });
        });

        it('converts unexpected errors into InternalServerErrorException', async () => {
            userService.create.mockRejectedValue(new Error('duplicate'));

            await expect(service.register(response, request, registerDto)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });

        it('preserves ConflictException thrown by UserService', async () => {
            const conflictError = new ConflictException('exists');
            userService.create.mockRejectedValue(conflictError);

            await expect(service.register(response, request, registerDto)).rejects.toBe(
                conflictError,
            );
        });
    });

    describe('login', () => {
        const baseUser = {
            id: 'user-id',
            name: 'John Doe',
            email: 'john.doe@example.com',
            password: 'hash',
            isActive: true,
            role: UserRoles.USER,
        } as const;

        beforeEach(() => {
            compareMock.mockResolvedValue(true);
        });

        it('logs user in when credentials are valid and role is allowed', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );

            const result = await service.login(response, request, loginDto);

            expect(userService.findByLogin).toHaveBeenCalledWith(loginDto.login);
            expect(compareMock).toHaveBeenCalledWith(loginDto.password, baseUser.password);
            expect(deviceIdService.getOrCreateDeviceId).toHaveBeenCalledWith(request);
            expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(response, 'device-id');
            expect(authSessionService.issueTokensDeferred).toHaveBeenCalledWith(
                expect.objectContaining({
                    id: 'user-id',
                    sessionId: 'test-session-id',
                }),
                true,
                'test-transaction',
            );
            expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                role: UserRoles.USER,
            });
            expect(tokenService.revokeTokensBySession).toHaveBeenCalledWith(
                'test-session-id',
                'test-transaction',
            );
            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: baseUser.email,
                role: baseUser.role,
            });
        });

        it('reuses existing active session when reusable session exists', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            sessionService.findReusableSession.mockResolvedValue({
                id: 'existing-session-id',
                userId: 'user-id',
                realm: 'customer',
                deviceId: 'device-id',
                expiresAt: new Date(Date.now() + 10000),
                createdAt: new Date(),
                updatedAt: new Date(),
                revokedAt: null,
            });

            const result = await service.login(response, request, loginDto);

            expect(sessionService.reuseSession).toHaveBeenCalledWith(
                'existing-session-id',
                expect.any(Object),
                undefined,
                'test-transaction',
            );
            expect(tokenService.revokeTokensBySession).toHaveBeenCalledWith(
                'existing-session-id',
                'test-transaction',
            );
            expect(authSessionService.issueTokensDeferred).toHaveBeenCalledWith(
                expect.objectContaining({ sessionId: 'existing-session-id' }),
                true,
                'test-transaction',
            );
            expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                role: UserRoles.USER,
            });
            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: baseUser.email,
                role: baseUser.role,
            });
        });

        it('throws UnauthorizedException when password does not match', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            compareMock.mockResolvedValue(false);

            await expect(service.login(response, request, loginDto)).rejects.toBeInstanceOf(
                UnauthorizedException,
            );
        });

        it('throws UnauthorizedException when stored password hash is invalid', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            compareMock.mockRejectedValue(new Error('pchstr must contain a $ as first char'));

            await expect(service.login(response, request, loginDto)).rejects.toBeInstanceOf(
                UnauthorizedException,
            );
            expect(authSessionService.issueTokensDeferred).not.toHaveBeenCalled();
        });

        it('rejects when user role is outside allowed entry point roles', async () => {
            userService.findByLogin.mockResolvedValue({
                ...baseUser,
                role: UserRoles.PLATFORM_ADMIN,
            } as AwaitedReturn<ReturnType<UserService['findByLogin']>>);

            await expect(
                service.login(response, request, loginDto, [UserRoles.USER, UserRoles.USER]),
            ).rejects.toBeInstanceOf(UnauthorizedException);

            expect(authSessionService.issueTokens).not.toHaveBeenCalled();
            expect(authSessionService.issueTokensDeferred).not.toHaveBeenCalled();
        });

        it('converts unexpected repository errors into InternalServerErrorException', async () => {
            userService.findByLogin.mockRejectedValue(new Error('db down'));

            await expect(service.login(response, request, loginDto)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });

        it('records suspicious_device when risk decision is medium', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 30,
                decision: 'medium',
                reasons: ['new_device', 'new_location'],
                requiredAction: 'alert',
            });
            securityEventService.recordSuspiciousDevice = jest.fn().mockResolvedValue(null);

            await service.login(response, request, loginDto);

            expect(securityEventService.recordSuspiciousDevice).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    ipAddress: '203.0.113.1',
                    metadata: {
                        deviceId: 'device-id',
                        riskSignals: 'new_device,new_location',
                    },
                },
                'test-transaction',
            );
        });

        it('records login_approval_required shadow event when risk decision is high and gate is disabled', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 60,
                decision: 'high',
                reasons: ['impossible_travel'],
                requiredAction: 'challenge',
            });
            configService.get.mockReturnValue(false);
            securityEventService.recordLoginApprovalRequired = jest.fn().mockResolvedValue(null);

            await service.login(response, request, loginDto);

            expect(securityEventService.recordLoginApprovalRequired).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    metadata: {
                        loginChallengeId: expect.any(String) as string,
                        deviceId: 'device-id',
                        enforcementEnabled: false,
                    },
                },
                'test-transaction',
            );
        });

        it('records suspicious_device, sends alert, and allows login when risk decision is critical and gate is disabled', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 90,
                decision: 'critical',
                reasons: ['device_binding_failed', 'high_failed_login_attempts'],
                requiredAction: 'deny',
            });
            configService.get.mockReturnValue(false);
            securityEventService.recordSuspiciousDevice = jest.fn().mockResolvedValue(null);

            const result = await service.login(response, request, loginDto);

            expect(securityEventService.recordSuspiciousDevice).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    ipAddress: '203.0.113.1',
                    metadata: {
                        deviceId: 'device-id',
                        riskSignals: 'device_binding_failed,high_failed_login_attempts',
                    },
                },
                'test-transaction',
            );
            expect(mailTemplateService.sendSecurityAlert).toHaveBeenCalled();
            expect(securityEventService.recordLoginFailed).not.toHaveBeenCalled();
            expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                role: UserRoles.USER,
            });
            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: baseUser.email,
                role: baseUser.role,
            });
        });

        it('creates a challenge, records login_approval_required, does not issue tokens, and returns CheckpointResponse when risk is high and gate is enabled', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 60,
                decision: 'high',
                reasons: ['impossible_travel'],
                requiredAction: 'challenge',
            });
            configService.get.mockReturnValue(true);
            securityEventService.recordLoginApprovalRequired = jest.fn().mockResolvedValue(null);

            const result = await service.login(response, request, loginDto);

            expect(loginChallengeService.createLoginChallenge).toHaveBeenCalledWith(
                'user-id',
                'customer',
                'test-known-device-id',
                'device-id',
                expect.any(Object),
                {
                    riskScore: 60,
                    riskReason: 'impossible_travel',
                },
                expect.any(Date),
                'test-transaction',
            );

            expect(securityEventService.recordLoginApprovalRequired).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    metadata: {
                        loginChallengeId: 'mock-challenge-uuid',
                        deviceId: 'device-id',
                        enforcementEnabled: true,
                    },
                },
                'test-transaction',
            );

            expect(authSessionService.issueTokens).not.toHaveBeenCalled();
            expect(authSessionService.issueTokensDeferred).not.toHaveBeenCalled();
            expect(sessionService.createSession).not.toHaveBeenCalled();
            expect(sessionService.reuseSession).not.toHaveBeenCalled();
            expect(result).toEqual({
                checkpointRequired: true,
                loginChallengeId: 'mock-challenge-uuid',
                checkpointToken: 'mock-checkpoint-token',
                expiresInSeconds: 300,
                resendAvailableInSeconds: 60,
            });
        });

        it('does not issue auth state when checkpoint email cannot be queued', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 60,
                decision: 'high',
                reasons: ['impossible_travel'],
                requiredAction: 'challenge',
            });
            configService.get.mockReturnValue(true);
            mailTemplateService.sendVerificationCodeOrThrow.mockRejectedValue(
                new ServiceUnavailableException('Verification email could not be queued.'),
            );

            await expect(service.login(response, request, loginDto)).rejects.toThrow(
                'Verification email could not be queued.',
            );

            expect(loginChallengeService.createLoginChallenge).toHaveBeenCalled();
            expect(mailTemplateService.sendVerificationCodeOrThrow).toHaveBeenCalledWith(
                baseUser.email,
                baseUser.name,
                '123456',
                5,
                'test-transaction',
            );
            expect(authSessionService.issueTokens).not.toHaveBeenCalled();
            expect(authSessionService.issueTokensDeferred).not.toHaveBeenCalled();
            expect(sessionService.createSession).not.toHaveBeenCalled();
            expect(sessionService.reuseSession).not.toHaveBeenCalled();
            expect(authSessionService.applyTokens).not.toHaveBeenCalled();
        });

        it('records suspicious_device, sends alert, and allows login when risk is critical and gate is enabled', async () => {
            userService.findByLogin.mockResolvedValue(
                baseUser as AwaitedReturn<ReturnType<UserService['findByLogin']>>,
            );
            riskPolicyService.evaluateRisk.mockReturnValue({
                score: 90,
                decision: 'critical',
                reasons: ['device_binding_failed', 'high_failed_login_attempts'],
                requiredAction: 'deny',
            });
            configService.get.mockReturnValue(true);
            securityEventService.recordSuspiciousDevice = jest.fn().mockResolvedValue(null);

            const result = await service.login(response, request, loginDto);

            expect(securityEventService.recordSuspiciousDevice).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    ipAddress: '203.0.113.1',
                    metadata: {
                        deviceId: 'device-id',
                        riskSignals: 'device_binding_failed,high_failed_login_attempts',
                    },
                },
                'test-transaction',
            );

            expect(mailTemplateService.sendSecurityAlert).toHaveBeenCalled();
            expect(securityEventService.recordLoginFailed).not.toHaveBeenCalled();
            expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                role: UserRoles.USER,
            });
            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: baseUser.email,
                role: baseUser.role,
            });
        });
    });

    describe('verifyLoginCheckpoint', () => {
        const verifyDto = {
            checkpointToken: 'mock-checkpoint-token',
            code: '123456',
        };

        const mockChallenge = {
            id: 'mock-challenge-uuid',
            userId: 'user-id',
            realm: 'customer',
            knownDeviceId: 'test-known-device-id',
            deviceId: 'test-device-id',
            expiresAt: new Date(Date.now() + 100000),
            consumedAt: null,
            failedAt: null,
            expiredAt: null,
            attemptCount: 0,
            maxAttempts: 3,
        };

        const mockUser = {
            id: 'user-id',
            name: 'John Doe',
            email: 'john@example.com',
            role: UserRoles.USER,
            isActive: true,
        };

        beforeEach(() => {
            userService.findById = jest.fn().mockResolvedValue(mockUser);
            knownDeviceService.trustKnownDevice = jest.fn().mockResolvedValue({});
            securityEventService.recordLoginApprovalPassed = jest.fn().mockResolvedValue(null);
            securityEventService.recordLoginApprovalFailed = jest.fn().mockResolvedValue(null);
            loginChallengeService.getChallenge.mockResolvedValue(mockChallenge);
            loginChallengeService.verifyLoginChallenge.mockResolvedValue(true);
            loginChallengeService.approveAndConsumeLoginChallengeAtomically.mockImplementation(
                async (_challengeId: string, issueAuthState: (tx: any) => Promise<unknown>) => {
                    return await issueAuthState('test-transaction');
                },
            );
        });

        it('throws UnauthorizedException when challenge is not found', async () => {
            loginChallengeService.getChallenge.mockResolvedValue(null);

            await expect(
                service.verifyLoginCheckpoint(response, request, verifyDto),
            ).rejects.toThrow(new UnauthorizedException('Invalid or expired login challenge.'));
        });

        it('records login_approval_failed and throws UnauthorizedException when verification fails', async () => {
            loginChallengeService.verifyLoginChallenge.mockResolvedValue(false);

            await expect(
                service.verifyLoginCheckpoint(response, request, verifyDto),
            ).rejects.toThrow(new UnauthorizedException('Invalid or expired login code.'));

            expect(securityEventService.recordLoginApprovalFailed).toHaveBeenCalledWith({
                userId: 'user-id',
                realm: 'customer',
                metadata: {
                    loginChallengeId: 'mock-challenge-uuid',
                    deviceId: 'test-device-id',
                    failureReason: 'invalid_code_or_challenge_state',
                },
            });
        });

        it('throws UnauthorizedException when user is not found or inactive', async () => {
            userService.findById.mockResolvedValue(null);

            await expect(
                service.verifyLoginCheckpoint(response, request, verifyDto),
            ).rejects.toThrow(new UnauthorizedException('User not found or inactive.'));
        });

        it('trusts device, creates session, issues tokens and logs success on successful verification', async () => {
            const result = await service.verifyLoginCheckpoint(response, request, verifyDto);

            expect(knownDeviceService.trustKnownDevice).toHaveBeenCalledWith(
                'test-known-device-id',
                null,
                'test-transaction',
                expect.any(Date),
            );

            expect(sessionService.createSession).toHaveBeenCalledWith(
                'user-id',
                'customer',
                'test-known-device-id',
                'test-device-id',
                expect.any(Object),
                'test-transaction',
            );

            expect(tokenService.revokeTokensBySession).toHaveBeenCalledWith(
                'test-session-id',
                'test-transaction',
            );

            expect(authSessionService.issueTokensDeferred).toHaveBeenCalledWith(
                expect.objectContaining({
                    id: 'user-id',
                    sessionId: 'test-session-id',
                }),
                true,
                'test-transaction',
            );
            expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(
                response,
                'test-device-id',
            );
            expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                accessToken: 'deferred-access-token',
                refreshToken: 'deferred-refresh-token',
                role: UserRoles.USER,
            });

            expect(securityEventService.recordLoginApprovalPassed).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    sessionId: 'test-session-id',
                    knownDeviceId: 'test-known-device-id',
                    metadata: {
                        loginChallengeId: 'mock-challenge-uuid',
                        deviceId: 'test-device-id',
                    },
                },
                'test-transaction',
            );

            expect(result).toEqual({
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            });
        });

        it('throws UnauthorizedException when atomic consumption returns null', async () => {
            loginChallengeService.approveAndConsumeLoginChallengeAtomically.mockResolvedValue(null);

            await expect(
                service.verifyLoginCheckpoint(response, request, verifyDto),
            ).rejects.toThrow(new UnauthorizedException('Invalid or expired login challenge.'));
        });
    });

    describe('resendLoginCheckpoint', () => {
        const resendDto = {
            checkpointToken: 'old-checkpoint-token',
        };

        const mockChallenge = {
            id: 'old-challenge-uuid',
            userId: 'user-id',
            realm: 'customer',
            knownDeviceId: 'known-device-id',
            deviceId: 'original-device-id',
            expiresAt: new Date(Date.now() + 100000),
            consumedAt: null,
            failedAt: null,
            expiredAt: null,
            attemptCount: 0,
            maxAttempts: 5,
            createdAt: new Date(Date.now() - 60_000),
            riskScore: 60,
            riskReason: 'impossible_travel',
        };

        const mockUser = {
            id: 'user-id',
            name: 'John Doe',
            email: 'john@example.com',
            role: UserRoles.USER,
            isActive: true,
        };

        beforeEach(() => {
            userService.findById.mockResolvedValue(mockUser);
            loginChallengeService.getChallenge.mockResolvedValue(mockChallenge);
            loginChallengeService.getLatestChallengeForDevice.mockResolvedValue(mockChallenge);
            loginChallengeService.getResendEligibility.mockReturnValue({
                eligible: true,
                state: 'active',
                resendWindowExpiresAt: new Date(Date.now() + 100000),
            });
            loginChallengeService.createLoginChallenge.mockResolvedValue({
                challenge: { id: 'new-challenge-uuid' },
                checkpointToken: 'new-checkpoint-token',
                code: '654321',
            });
        });

        it('creates a new challenge, sends strict email and returns a new checkpoint token', async () => {
            const result = await service.resendLoginCheckpoint(response, request, resendDto);

            expect(loginChallengeService.getChallenge).toHaveBeenCalledWith('old-checkpoint-token');
            expect(loginChallengeService.getLatestChallengeForDevice).toHaveBeenCalledWith(
                'user-id',
                'customer',
                'original-device-id',
            );
            expect(loginChallengeResendPolicyService.reserveResendAttempt).toHaveBeenCalledWith({
                userId: 'user-id',
                realm: 'customer',
                deviceId: 'original-device-id',
                ipAddress: '203.0.113.1',
            });
            expect(loginChallengeService.createLoginChallenge).toHaveBeenCalledWith(
                'user-id',
                'customer',
                'known-device-id',
                'original-device-id',
                {
                    ipAddress: '203.0.113.1',
                    userAgent: 'Jest',
                    country: 'US',
                    region: 'CA',
                    city: 'San Jose',
                },
                {
                    riskScore: 60,
                    riskReason: 'impossible_travel',
                },
                expect.any(Date),
                'test-transaction',
            );
            expect(mailTemplateService.sendVerificationCodeOrThrow).toHaveBeenCalledWith(
                'john@example.com',
                'John Doe',
                '654321',
                5,
                'test-transaction',
            );
            expect(securityEventService.recordLoginApprovalResent).toHaveBeenCalledWith(
                {
                    userId: 'user-id',
                    realm: 'customer',
                    ipAddress: '203.0.113.1',
                    userAgent: 'Jest',
                    metadata: {
                        oldLoginChallengeId: 'old-challenge-uuid',
                        newLoginChallengeId: 'new-challenge-uuid',
                        deviceId: 'original-device-id',
                    },
                },
                'test-transaction',
            );
            expect(securityEventService.recordLoginApprovalResendFailed).not.toHaveBeenCalled();
            expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(
                response,
                'original-device-id',
            );
            expect(result).toEqual({
                checkpointRequired: true,
                loginChallengeId: 'new-challenge-uuid',
                checkpointToken: 'new-checkpoint-token',
                expiresInSeconds: 300,
                resendAvailableInSeconds: 60,
            });
        });

        it('allows resend for expired but eligible challenge inside resend window', async () => {
            loginChallengeService.getResendEligibility.mockReturnValue({
                eligible: true,
                state: 'expired',
                resendWindowExpiresAt: new Date(Date.now() + 100000),
            });

            await expect(
                service.resendLoginCheckpoint(response, request, resendDto),
            ).resolves.toEqual(
                expect.objectContaining({
                    checkpointToken: 'new-checkpoint-token',
                }),
            );
        });

        it('rejects ineligible challenge before reserving resend policy', async () => {
            loginChallengeService.getResendEligibility.mockReturnValue({
                eligible: false,
                reason: 'outside_resend_window',
                resendWindowExpiresAt: new Date(Date.now() - 1000),
            });

            await expect(
                service.resendLoginCheckpoint(response, request, resendDto),
            ).rejects.toThrow(new UnauthorizedException('Invalid or expired login challenge.'));

            expect(loginChallengeResendPolicyService.reserveResendAttempt).not.toHaveBeenCalled();
            expect(loginChallengeService.createLoginChallenge).not.toHaveBeenCalled();
            expect(securityEventService.recordLoginApprovalResendFailed).toHaveBeenCalledWith({
                userId: 'user-id',
                realm: 'customer',
                ipAddress: '203.0.113.1',
                userAgent: 'Jest',
                metadata: {
                    oldLoginChallengeId: 'old-challenge-uuid',
                    deviceId: 'original-device-id',
                    failureReason: 'ineligible_outside_resend_window',
                },
            });
        });

        it('rejects rate-limited resend before creating a new challenge', async () => {
            const setHeader = jest.fn();
            const responseWithHeader = {
                ...response,
                setHeader,
            } as unknown as Response;
            loginChallengeResendPolicyService.reserveResendAttempt.mockResolvedValue({
                allowed: false,
                reason: 'cooldown',
                retryAfterSeconds: 42,
            });

            await expect(
                service.resendLoginCheckpoint(responseWithHeader, request, resendDto),
            ).rejects.toMatchObject(
                new HttpException(
                    'Too many verification code requests.',
                    HttpStatus.TOO_MANY_REQUESTS,
                ),
            );

            expect(setHeader).toHaveBeenCalledWith('Retry-After', '42');
            expect(loginChallengeService.createLoginChallenge).not.toHaveBeenCalled();
            expect(mailTemplateService.sendVerificationCodeOrThrow).not.toHaveBeenCalled();
            expect(securityEventService.recordLoginApprovalResendFailed).toHaveBeenCalledWith({
                userId: 'user-id',
                realm: 'customer',
                ipAddress: '203.0.113.1',
                userAgent: 'Jest',
                metadata: {
                    oldLoginChallengeId: 'old-challenge-uuid',
                    deviceId: 'original-device-id',
                    failureReason: 'rate_limited_cooldown',
                },
            });
        });
    });

    describe('logout', () => {
        it('revokes current session and clears cookies after successful revoke', async () => {
            const logoutRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'access-token',
                    [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                },
            });

            tokenService.getTokenData.mockReturnValue({
                userId: 'user-id',
                sessionId: 'session-id',
                jti: 'access-jti',
                role: UserRoles.USER,
            });
            tokenService.verifyRefreshToken.mockResolvedValue({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                user: {
                    id: 'user-id',
                    name: 'John',
                    email: 'john@example.com',
                    role: UserRoles.USER,
                },
            });

            const result = await service.logout(logoutRequest, response, UserRoles.USER);

            expect(result).toBe(true);
            expect(tokenService.getTokenData).toHaveBeenCalledWith('access-token');
            expect(tokenService.verifyRefreshToken).toHaveBeenCalledWith(
                'refresh-token',
                'customer',
            );
            expect(sessionService.revokeCurrentSession).toHaveBeenCalledWith({
                sessionId: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                accessTokenJti: 'access-jti',
                ipAddress: '203.0.113.1',
                userAgent: 'Jest',
            });
            expect(authSessionService.clearTokens).toHaveBeenCalledWith(
                response,
                UserRoles.USER,
            );
            expect(sessionService.revokeCurrentSession.mock.invocationCallOrder[0]).toBeLessThan(
                authSessionService.clearTokens.mock.invocationCallOrder[0],
            );
            expect(tokenService.revokeAccessSession).not.toHaveBeenCalled();
            expect(tokenService.removeRefreshToken).not.toHaveBeenCalled();
        });

        it('does not clear cookies when no current session can be resolved', async () => {
            const logoutRequest = createRequest({ cookies: {} });

            const result = await service.logout(logoutRequest, response, UserRoles.USER);

            expect(result).toBe(false);
            expect(authSessionService.clearTokens).not.toHaveBeenCalled();
            expect(tokenService.verifyRefreshToken).not.toHaveBeenCalled();
            expect(tokenService.revokeAccessSession).not.toHaveBeenCalled();
            expect(tokenService.removeRefreshToken).not.toHaveBeenCalled();
            expect(sessionService.revokeCurrentSession).not.toHaveBeenCalled();
        });

        it('does not clear cookies when refresh token verification fails and access token is absent', async () => {
            const logoutRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'expired-refresh-token',
                },
            });

            const expiredTokenError = new Error('jwt expired');
            expiredTokenError.name = 'TokenExpiredError';
            tokenService.verifyRefreshToken.mockRejectedValue(expiredTokenError);

            const result = await service.logout(logoutRequest, response, UserRoles.USER);

            expect(result).toBe(false);
            expect(authSessionService.clearTokens).not.toHaveBeenCalled();
            expect(tokenService.removeRefreshToken).not.toHaveBeenCalled();
            expect(sessionService.revokeCurrentSession).not.toHaveBeenCalled();
        });

        it('rethrows unexpected refresh token verification errors', async () => {
            const logoutRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                },
            });
            const unexpectedError = new Error('database unavailable');

            tokenService.verifyRefreshToken.mockRejectedValue(unexpectedError);

            await expect(service.logout(logoutRequest, response, UserRoles.USER)).rejects.toBe(
                unexpectedError,
            );
            expect(authSessionService.clearTokens).not.toHaveBeenCalled();
            expect(tokenService.removeRefreshToken).not.toHaveBeenCalled();
            expect(sessionService.revokeCurrentSession).not.toHaveBeenCalled();
        });

        it('does not clear cookies if current-session revoke fails', async () => {
            const logoutRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                },
            });
            const revokeError = new Error('redis unavailable');

            tokenService.verifyRefreshToken.mockResolvedValue({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                user: {
                    id: 'user-id',
                    name: 'John',
                    email: 'john@example.com',
                    role: UserRoles.USER,
                },
            });
            sessionService.revokeCurrentSession.mockRejectedValue(revokeError);

            await expect(service.logout(logoutRequest, response, UserRoles.USER)).rejects.toBe(
                revokeError,
            );

            expect(authSessionService.clearTokens).not.toHaveBeenCalled();
        });
    });

    describe('getMe', () => {
        it('returns decoded payload when access token exists', () => {
            const meRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'access-token',
                },
            });

            tokenService.getTokenData.mockReturnValue({
                userId: 'user-id',
                sessionId: 'session-id',
                email: 'john@example.com',
                username: 'John',
                role: UserRoles.USER,
            } as ReturnType<AuthTokenService['getTokenData']>);

            const result = service.getMe(meRequest, UserRoles.USER);

            expect(result).toEqual({
                id: 'user-id',
                email: 'john@example.com',
                name: 'John',
                role: UserRoles.USER,
            });
        });

        it('throws UnauthorizedException when access token is missing', () => {
            expect(() => service.getMe(request, UserRoles.USER)).toThrow(UnauthorizedException);
        });
    });

    describe('refreshAccessToken', () => {
        it('re-issues access token when refresh token is valid', async () => {
            const refreshRequest = createRequest({
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                },
            });

            tokenService.verifyRefreshToken.mockResolvedValue({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                isGrace: false,
                user: {
                    id: 'user-id',
                    name: 'John Doe',
                    email: 'john.doe@example.com',
                    role: UserRoles.USER,
                },
            });

            const result = await service.refreshAccessToken(
                response,
                refreshRequest,
                COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
            );

            expect(result).toBe(true);
            expect(deviceIdService.getOrCreateDeviceId).toHaveBeenCalledWith(refreshRequest);
            expect(deviceIdService.setDeviceIdCookie).toHaveBeenCalledWith(response, 'device-id');
            expect(authSessionService.issueTokens).toHaveBeenCalledWith(
                response,
                expect.objectContaining({
                    id: 'user-id',
                    sessionId: 'session-id',
                    ipAddress: '203.0.113.1',
                    userAgent: 'Jest',
                }),
                false,
            );
            expect(tokenService.verifyRefreshToken).toHaveBeenCalledWith(
                'refresh-token',
                'customer',
            );
        });

        it('returns false when refresh token cookie is missing', async () => {
            const result = await service.refreshAccessToken(response, request, 'missing');

            expect(result).toBe(false);
            expect(tokenService.verifyRefreshToken).not.toHaveBeenCalled();
        });

        it('returns false when verification fails', async () => {
            const refreshRequest = createRequest({
                cookies: {
                    any: 'token',
                },
            });

            tokenService.verifyRefreshToken.mockResolvedValue({ verified: false });

            const result = await service.refreshAccessToken(response, refreshRequest, 'any');

            expect(result).toBe(false);
            expect(tokenService.verifyRefreshToken).toHaveBeenCalledWith('token', undefined);
            expect(authSessionService.issueTokens).not.toHaveBeenCalled();
        });

        describe('Rotation Integration and Grace Window Validation', () => {
            let refreshRequest: Request;

            beforeEach(() => {
                refreshRequest = createRequest({
                    cookies: {
                        [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                    },
                });

                tokenService.verifyRefreshToken.mockResolvedValue({
                    verified: true,
                    tokenId: 'refresh-token-id',
                    sessionId: 'session-id',
                    isGrace: false,
                    user: {
                        id: 'user-id',
                        name: 'John Doe',
                        email: 'john.doe@example.com',
                        role: UserRoles.USER,
                    },
                });

                sessionService.getSession.mockResolvedValue({
                    id: 'session-id',
                    userId: 'user-id',
                    realm: 'customer',
                    knownDeviceId: 'known-device-id',
                    deviceId: 'device-id',
                });
            });

            it('rotates refresh token and issues access token when REFRESH_ROTATION_ENABLED=true and token is current', async () => {
                configService.get.mockReturnValue(true); // REFRESH_ROTATION_ENABLED = true

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(true);
                expect(tokenService.rotateRefreshTokenById).toHaveBeenCalledWith(
                    'refresh-token-id',
                    expect.any(Object),
                );
                expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                    accessToken: 'new-access-token',
                    refreshToken: 'new-refresh-token',
                    role: UserRoles.USER,
                });
            });

            it('does not rotate and only issues access token when REFRESH_ROTATION_ENABLED=false', async () => {
                configService.get.mockReturnValue(false); // REFRESH_ROTATION_ENABLED = false

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(true);
                expect(tokenService.rotateRefreshTokenById).not.toHaveBeenCalled();
                expect(authSessionService.issueTokens).toHaveBeenCalledWith(
                    response,
                    expect.any(Object),
                    false,
                );
            });

            it('re-sends replacement refresh token when token is in grace window (isGrace=true)', async () => {
                configService.get.mockReturnValue(true); // REFRESH_ROTATION_ENABLED = true
                tokenService.verifyRefreshToken.mockResolvedValue({
                    verified: true,
                    tokenId: 'refresh-token-id',
                    sessionId: 'session-id',
                    isGrace: true,
                    user: {
                        id: 'user-id',
                        name: 'John Doe',
                        email: 'john.doe@example.com',
                        role: UserRoles.USER,
                    },
                });

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(true);
                expect(tokenService.rotateRefreshTokenById).not.toHaveBeenCalled();
                expect(tokenService.acceptReplacedTokenInGraceById).toHaveBeenCalledWith(
                    'refresh-token-id',
                );
                expect(authSessionService.applyTokens).toHaveBeenCalledWith(response, {
                    accessToken: 'new-access-token',
                    refreshToken: 'replacement-refresh-token',
                    role: UserRoles.USER,
                });
            });

            it('returns false when token is in grace window but REFRESH_ROTATION_ENABLED=false', async () => {
                configService.get.mockReturnValue(false); // REFRESH_ROTATION_ENABLED = false
                tokenService.verifyRefreshToken.mockResolvedValue({
                    verified: true,
                    tokenId: 'refresh-token-id',
                    sessionId: 'session-id',
                    isGrace: true,
                    user: {
                        id: 'user-id',
                        name: 'John Doe',
                        email: 'john.doe@example.com',
                        role: UserRoles.USER,
                    },
                });

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(false);
                expect(securityEventService.recordRefreshFailed).toHaveBeenCalled();
            });

            it('records failed refresh security event when verifyRefreshToken throws an error', async () => {
                tokenService.verifyRefreshToken.mockRejectedValue(
                    new Error('JWT verification failed'),
                );

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(false);
                expect(securityEventService.recordRefreshFailed).toHaveBeenCalledWith(
                    expect.objectContaining({
                        realm: 'customer',
                        metadata: expect.objectContaining({
                            failureReason: 'JWT verification failed',
                        }) as unknown,
                    }),
                );
            });

            it('updates session and known device last seen timestamp on successful refresh', async () => {
                configService.get.mockReturnValue(true);

                const result = await service.refreshAccessToken(
                    response,
                    refreshRequest,
                    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                );

                expect(result).toBe(true);
                expect(sessionService.touchLastSeenAt).toHaveBeenCalledWith('session-id', 60);
                expect(knownDeviceService.touchKnownDevice).toHaveBeenCalledWith(
                    'known-device-id',
                    expect.objectContaining({
                        lastIpAddress: '203.0.113.1',
                        lastUserAgent: 'Jest',
                    }),
                    60,
                );
            });
        });

        describe('Device Binding Validation', () => {
            let refreshRequest: Request;

            beforeEach(() => {
                refreshRequest = createRequest({
                    cookies: {
                        [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'refresh-token',
                    },
                });

                tokenService.verifyRefreshToken.mockResolvedValue({
                    verified: true,
                    tokenId: 'refresh-token-id',
                    sessionId: 'session-id',
                    isGrace: false,
                    user: {
                        id: 'user-id',
                        name: 'John Doe',
                        email: 'john.doe@example.com',
                        role: UserRoles.USER,
                    },
                });

                sessionService.getSession.mockResolvedValue({
                    id: 'session-id',
                    userId: 'user-id',
                    realm: 'customer',
                    knownDeviceId: 'known-device-id',
                    deviceId: 'device-id',
                });
            });

            describe('Shadow Mode (Enforcement Disabled)', () => {
                beforeEach(() => {
                    configService.get.mockReturnValue(false); // DEVICE_BINDING_REQUIRED_CUSTOMER = false
                });

                it('succeeds when deviceId cookie is missing', async () => {
                    deviceIdService.readDeviceId.mockReturnValue(null);

                    const result = await service.refreshAccessToken(
                        response,
                        refreshRequest,
                        COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                    );

                    expect(result).toBe(true);
                });

                it('succeeds when deviceId cookie is mismatched', async () => {
                    deviceIdService.readDeviceId.mockReturnValue('mismatched-device-id');

                    const result = await service.refreshAccessToken(
                        response,
                        refreshRequest,
                        COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                    );

                    expect(result).toBe(true);
                });
            });

            describe('Enforcement Mode (Enforcement Enabled)', () => {
                beforeEach(() => {
                    configService.get.mockReturnValue(true); // DEVICE_BINDING_REQUIRED_CUSTOMER = true
                });

                it('fails when deviceId cookie is missing', async () => {
                    deviceIdService.readDeviceId.mockReturnValue(null);

                    await expect(
                        service.refreshAccessToken(
                            response,
                            refreshRequest,
                            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                        ),
                    ).rejects.toBeInstanceOf(UnauthorizedException);

                    expect(securityEventService.recordRefreshFailed).toHaveBeenCalledWith(
                        expect.objectContaining({
                            realm: 'customer',
                            metadata: expect.objectContaining({
                                failureReason: 'Device binding mismatch: deviceId cookie missing',
                            }) as unknown,
                        }),
                    );
                    expect(authSessionService.issueTokens).not.toHaveBeenCalled();
                    expect(authSessionService.applyTokens).not.toHaveBeenCalled();
                });

                it('fails when deviceId cookie is mismatched', async () => {
                    deviceIdService.readDeviceId.mockReturnValue('mismatched-device-id');

                    await expect(
                        service.refreshAccessToken(
                            response,
                            refreshRequest,
                            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                        ),
                    ).rejects.toBeInstanceOf(UnauthorizedException);

                    expect(securityEventService.recordRefreshFailed).toHaveBeenCalledWith(
                        expect.objectContaining({
                            realm: 'customer',
                            metadata: expect.objectContaining({
                                failureReason: expect.stringContaining(
                                    'Device binding mismatch: cookie deviceId',
                                ) as unknown,
                            }) as unknown,
                        }),
                    );
                    expect(authSessionService.issueTokens).not.toHaveBeenCalled();
                    expect(authSessionService.applyTokens).not.toHaveBeenCalled();
                });

                it('uses admin device binding flag for admin refresh tokens', async () => {
                    const adminRefreshRequest = createRequest({
                        cookies: {
                            [COOKIE_NAMES.ADMIN_REFRESH_TOKEN]: 'admin-refresh-token',
                        },
                    });
                    configService.get.mockImplementation((key: string) => {
                        if (key === 'DEVICE_BINDING_REQUIRED_ADMIN') {
                            return true;
                        }
                        return false;
                    });
                    tokenService.verifyRefreshToken.mockResolvedValue({
                        verified: true,
                        tokenId: 'admin-refresh-token-id',
                        sessionId: 'admin-session-id',
                        isGrace: false,
                        user: {
                            id: 'admin-user-id',
                            name: 'Admin',
                            email: 'admin@example.com',
                            role: UserRoles.PLATFORM_ADMIN,
                        },
                    });
                    sessionService.getSession.mockResolvedValue({
                        id: 'admin-session-id',
                        userId: 'admin-user-id',
                        realm: 'admin',
                        knownDeviceId: 'admin-known-device-id',
                        deviceId: 'admin-device-id',
                    });
                    deviceIdService.readDeviceId.mockReturnValue('mismatched-device-id');

                    await expect(
                        service.refreshAccessToken(
                            response,
                            adminRefreshRequest,
                            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
                        ),
                    ).rejects.toBeInstanceOf(UnauthorizedException);

                    expect(tokenService.verifyRefreshToken).toHaveBeenCalledWith(
                        'admin-refresh-token',
                        'admin',
                    );
                    expect(configService.get).toHaveBeenCalledWith('DEVICE_BINDING_REQUIRED_ADMIN');
                    expect(securityEventService.recordRefreshFailed).toHaveBeenCalledWith(
                        expect.objectContaining({
                            userId: 'admin-user-id',
                            realm: 'admin',
                            metadata: expect.objectContaining({
                                failureReason: expect.stringContaining(
                                    'Device binding mismatch: cookie deviceId',
                                ) as unknown,
                            }) as unknown,
                        }),
                    );
                    expect(authSessionService.issueTokens).not.toHaveBeenCalled();
                    expect(authSessionService.applyTokens).not.toHaveBeenCalled();
                });

                it('succeeds when deviceId cookie matches session deviceId', async () => {
                    deviceIdService.readDeviceId.mockReturnValue('device-id');

                    const result = await service.refreshAccessToken(
                        response,
                        refreshRequest,
                        COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
                    );

                    expect(result).toBe(true);
                });
            });
        });
    });
});
