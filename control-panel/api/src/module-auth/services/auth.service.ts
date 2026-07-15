import {
    HttpException,
    Injectable,
    InternalServerErrorException,
    UnauthorizedException,
    Logger,
    HttpStatus,
} from '@nestjs/common';
import { UserService } from '@/module-user/services/user.service';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { UserJwtPayload, User, CheckpointResponse } from '@/module-user/interfaces/user.interfaces';
import { RegisterUserDto } from '@/module-auth/dto/register-user.dto';
import type { Request, Response } from 'express';
import { LoginUserDto } from '@/module-auth/dto/login-user.dto';
import { VerifyCheckpointDto } from '@/module-auth/dto/verify-checkpoint.dto';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { COOKIE_NAMES, UserRoles, deriveAuthRealmFromRole } from '@/module-auth/enums/auth.enums';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { SessionService } from '@/module-auth-token/services/session.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import {
    TransactionRepository,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import { type AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { RefreshTokenVerificationResult } from '@/module-auth-token/types/auth-token.types';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { UserModel } from '@/module-user/types/user.types';
import { ConfigService } from '@nestjs/config';
import {
    RiskPolicyService,
    RiskEvaluationContext,
    UserSecurityHistory,
    RiskDecision,
} from '@/module-auth-token/services/risk-policy.service';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { AuthRealm } from '@/module-auth/enums/auth.enums';
import { randomUUID } from 'crypto';
import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import type { LoginChallengeSelect } from '@/module-auth-token/schemas/login-challenges.schema';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { parseUserAgent } from '@/common/utils/request-metadata.util';
import {
    LOGIN_CHALLENGE_RESEND_COOLDOWN_SECONDS,
    LOGIN_CHALLENGE_TTL_MINUTES,
    LOGIN_CHALLENGE_TTL_SECONDS,
} from '@/module-auth-token/constants/login-challenge.constants';
import { LoginChallengeResendPolicyService } from '@/module-auth-token/services/login-challenge-resend-policy.service';

type DeferredAuthTokens = {
    accessToken: string;
    refreshToken: string | null;
    role: UserRoles;
};

type RequestCookies = Record<string, string | undefined>;

@Injectable()
export class AuthService {
    private readonly logger = new Logger(AuthService.name);

    public constructor(
        private readonly userService: UserService,
        private readonly tokenService: AuthTokenService,
        public readonly authSessionService: AuthSessionService,
        private readonly deviceIdService: DeviceIdService,
        private readonly knownDeviceService: KnownDeviceService,
        private readonly sessionService: SessionService,
        private readonly securityEventService: SecurityEventService,
        private readonly geoIpService: GeoIpService,
        private readonly transactionRepository: TransactionRepository,
        private readonly configService: ConfigService,
        private readonly riskPolicyService: RiskPolicyService,
        private readonly securityEventRepository: SecurityEventRepository,
        private readonly loginChallengeService: LoginChallengeService,
        private readonly mailTemplateService: MailTemplateService,
        private readonly loginChallengeResendPolicyService: LoginChallengeResendPolicyService,
    ) {}

    async register(
        response: Response,
        request: Request,
        registerUserDto: RegisterUserDto,
    ): Promise<User> {
        try {
            const { id, name, email, role } = await this.userService.create(registerUserDto);
            const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
            const userJWTPayload = {
                id,
                name,
                email,
                role,
                ipAddress,
                userAgent,
            };

            const { deviceId } = this.deviceIdService.getOrCreateDeviceId(request);
            this.deviceIdService.setDeviceIdCookie(response, deviceId);

            await this.authSessionService.issueTokens(response, userJWTPayload);

            return {
                id,
                name,
                email,
                role,
            };
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to register user');
        }
    }

    async login(
        response: Response,
        request: Request,
        loginUserDto: LoginUserDto,
        allowedRoles?: UserRoles[],
    ): Promise<User | CheckpointResponse> {
        try {
            const { login, password } = loginUserDto;
            const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
            const user: UserModel | null = await this.userService.findByLogin(login);

            if (!user || !user.isActive) {
                throw new UnauthorizedException('Login or password is not valid.');
            }

            if (allowedRoles && !allowedRoles.includes(user.role)) {
                throw new UnauthorizedException('Login or password is not valid.');
            }

            const isPasswordValid = await this.verifyPassword(password, user.password);
            if (!isPasswordValid) {
                throw new UnauthorizedException('Login or password is not valid.');
            }

            const userId = user.id;
            const realm = deriveAuthRealmFromRole(user.role);

            const { deviceId } = this.deviceIdService.getOrCreateDeviceId(request);

            const geo = await this.geoIpService.lookup(ipAddress);

            const userPayload = {
                id: user.id,
                name: user.name,
                email: user.email,
                role: user.role,
            };

            const sessionMetadata = {
                ipAddress,
                lastCountry: geo.country,
                lastRegion: geo.region,
                lastCity: geo.city,
                userAgent,
            };

            const knownDeviceMetadata = {
                lastIpAddress: ipAddress,
                lastCountry: geo.country,
                lastRegion: geo.region,
                lastCity: geo.city,
                lastUserAgent: userAgent,
            };

            let sessionId: string;
            let checkpointResponse: CheckpointResponse | null = null;
            let deferredTokens: DeferredAuthTokens | null = null;

            await this.transactionRepository.run(async transaction => {
                const knownDevice = await this.knownDeviceService.findOrCreateKnownDevice(
                    userId,
                    realm,
                    deviceId,
                    knownDeviceMetadata,
                    transaction,
                );

                const currentMetadata = {
                    ipAddress,
                    userAgent,
                    country: geo.country ?? undefined,
                    region: geo.region ?? undefined,
                    city: geo.city ?? undefined,
                };

                const riskDecision = await this.evaluateLoginRisk(
                    user,
                    realm,
                    deviceId,
                    currentMetadata,
                    transaction,
                );
                const riskReason = riskDecision.reasons.join(',');
                const sessionMetadataWithRisk = {
                    ...sessionMetadata,
                    riskScore: riskDecision.score,
                    riskReason: riskReason || null,
                };

                const isCheckpointEnabled =
                    this.configService.get<boolean>('LOGIN_CHECKPOINT_ENABLED') ?? false;

                if (isCheckpointEnabled && riskDecision.decision === 'high') {
                    const { challenge, checkpointToken, code } =
                        await this.loginChallengeService.createLoginChallenge(
                            userId,
                            realm,
                            knownDevice.id,
                            deviceId,
                            currentMetadata,
                            {
                                riskScore: riskDecision.score,
                                riskReason,
                            },
                            new Date(),
                            transaction,
                        );

                    await this.securityEventService.recordLoginApprovalRequired(
                        {
                            userId,
                            realm,
                            metadata: {
                                loginChallengeId: challenge.id,
                                deviceId,
                                enforcementEnabled: true,
                            },
                        },
                        transaction,
                    );

                    await this.mailTemplateService.sendVerificationCodeOrThrow(
                        user.email,
                        user.name || 'User',
                        code,
                        LOGIN_CHALLENGE_TTL_MINUTES,
                        transaction,
                    );

                    checkpointResponse = {
                        checkpointRequired: true,
                        loginChallengeId: challenge.id,
                        checkpointToken,
                        expiresInSeconds: LOGIN_CHALLENGE_TTL_SECONDS,
                        resendAvailableInSeconds: LOGIN_CHALLENGE_RESEND_COOLDOWN_SECONDS,
                    };
                    return;
                }

                if (riskDecision.decision !== 'low') {
                    await this.securityEventService.recordSuspiciousDevice(
                        {
                            userId,
                            realm,
                            ipAddress,
                            metadata: {
                                deviceId,
                                riskSignals: riskReason,
                            },
                        },
                        transaction,
                    );

                    const uaParsed = parseUserAgent(userAgent);
                    const details = `Unrecognized login attempt detected. OS: ${uaParsed.os}, Browser: ${uaParsed.browser}. IP: ${ipAddress}, Location: ${geo.city || 'unknown'}, ${geo.country || 'unknown'}.`;
                    await this.mailTemplateService.sendSecurityAlert(
                        user.email,
                        user.name || 'User',
                        'Unrecognized Login',
                        details,
                        new Date().toISOString(),
                        transaction,
                    );
                }

                if (riskDecision.decision === 'high') {
                    await this.securityEventService.recordLoginApprovalRequired(
                        {
                            userId,
                            realm,
                            metadata: {
                                loginChallengeId: randomUUID(),
                                deviceId,
                                enforcementEnabled: false,
                            },
                        },
                        transaction,
                    );
                }

                const reusableSession = await this.sessionService.findReusableSession(
                    userId,
                    realm,
                    knownDevice.id,
                    transaction,
                );

                if (reusableSession) {
                    sessionId = reusableSession.id;
                    await this.sessionService.reuseSession(
                        sessionId,
                        sessionMetadataWithRisk,
                        undefined,
                        transaction,
                    );
                } else {
                    const session = await this.sessionService.createSession(
                        userId,
                        realm,
                        knownDevice.id,
                        deviceId,
                        sessionMetadataWithRisk,
                        transaction,
                    );
                    sessionId = session.id;
                }

                await this.tokenService.revokeTokensBySession(sessionId, transaction);

                const userJWTPayload: UserJwtPayload = {
                    ...userPayload,
                    sessionId,
                    ipAddress,
                    userAgent,
                };

                const issuedTokens = await this.authSessionService.issueTokensDeferred(
                    userJWTPayload,
                    true,
                    transaction,
                );

                deferredTokens = {
                    accessToken: issuedTokens.accessToken,
                    refreshToken: issuedTokens.refreshToken,
                    role: issuedTokens.payloadWithSession.role,
                };

                await this.securityEventService.recordLoginSuccess(
                    {
                        userId,
                        realm,
                        sessionId,
                        knownDeviceId: knownDevice.id,
                        ipAddress,
                        userAgent,
                    },
                    transaction,
                );
            });

            if (checkpointResponse) {
                this.deviceIdService.setDeviceIdCookie(response, deviceId);
                return checkpointResponse;
            }

            if (!deferredTokens) {
                throw new InternalServerErrorException('Login failed during token issuance.');
            }

            this.deviceIdService.setDeviceIdCookie(response, deviceId);
            this.authSessionService.applyTokens(response, deferredTokens);

            return userPayload;
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to log in user');
        }
    }

    async verifyLoginCheckpoint(
        response: Response,
        request: Request,
        verifyCheckpointDto: VerifyCheckpointDto,
    ): Promise<User> {
        try {
            const { checkpointToken, code } = verifyCheckpointDto;
            const now = new Date();

            const challenge = await this.loginChallengeService.getChallenge(checkpointToken);
            if (!challenge) {
                throw new UnauthorizedException('Invalid or expired login challenge.');
            }

            const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
            const geo = await this.geoIpService.lookup(ipAddress);

            const isValid = await this.loginChallengeService.verifyLoginChallenge(
                checkpointToken,
                code,
                now,
            );

            if (!isValid) {
                await this.securityEventService.recordLoginApprovalFailed({
                    userId: challenge.userId,
                    realm: challenge.realm as AuthRealm,
                    metadata: {
                        loginChallengeId: challenge.id,
                        deviceId: challenge.deviceId,
                        failureReason: 'invalid_code_or_challenge_state',
                    },
                });
                throw new UnauthorizedException('Invalid or expired login code.');
            }

            const sessionMetadata = {
                ipAddress,
                lastCountry: geo.country,
                lastRegion: geo.region,
                lastCity: geo.city,
                userAgent,
            };

            const user = await this.userService.findById(challenge.userId);
            if (!user || !user.isActive) {
                throw new UnauthorizedException('User not found or inactive.');
            }

            const userPayload = {
                id: user.id,
                name: user.name,
                email: user.email,
                role: user.role,
            };
            let deferredTokens: DeferredAuthTokens | null = null;

            const result =
                await this.loginChallengeService.approveAndConsumeLoginChallengeAtomically(
                    challenge.id,
                    async transaction => {
                        await this.knownDeviceService.trustKnownDevice(
                            challenge.knownDeviceId!,
                            null,
                            transaction,
                            now,
                        );

                        let sessionId: string;
                        const reusableSession = await this.sessionService.findReusableSession(
                            challenge.userId,
                            challenge.realm as AuthRealm,
                            challenge.knownDeviceId!,
                            transaction,
                        );

                        if (reusableSession) {
                            sessionId = reusableSession.id;
                            await this.sessionService.reuseSession(
                                sessionId,
                                sessionMetadata,
                                undefined,
                                transaction,
                            );
                        } else {
                            const session = await this.sessionService.createSession(
                                challenge.userId,
                                challenge.realm as AuthRealm,
                                challenge.knownDeviceId!,
                                challenge.deviceId,
                                sessionMetadata,
                                transaction,
                            );
                            sessionId = session.id;
                        }

                        await this.tokenService.revokeTokensBySession(sessionId, transaction);

                        const userJWTPayload: UserJwtPayload = {
                            ...userPayload,
                            sessionId,
                            ipAddress,
                            userAgent,
                        };

                        const issuedTokens = await this.authSessionService.issueTokensDeferred(
                            userJWTPayload,
                            true,
                            transaction,
                        );

                        deferredTokens = {
                            accessToken: issuedTokens.accessToken,
                            refreshToken: issuedTokens.refreshToken,
                            role: issuedTokens.payloadWithSession.role,
                        };

                        await this.securityEventService.recordLoginApprovalPassed(
                            {
                                userId: challenge.userId,
                                realm: challenge.realm as AuthRealm,
                                sessionId,
                                knownDeviceId: challenge.knownDeviceId!,
                                metadata: {
                                    loginChallengeId: challenge.id,
                                    deviceId: challenge.deviceId,
                                },
                            },
                            transaction,
                        );

                        return userPayload;
                    },
                    now,
                );

            if (!result) {
                throw new UnauthorizedException('Invalid or expired login challenge.');
            }

            if (!deferredTokens) {
                throw new InternalServerErrorException(
                    'Login checkpoint failed during token issuance.',
                );
            }

            this.deviceIdService.setDeviceIdCookie(response, challenge.deviceId);
            this.authSessionService.applyTokens(response, deferredTokens);

            return result;
        } catch (error) {
            this.handleUnexpectedError(error, 'Failed to verify login challenge');
        }
    }

    async resendLoginCheckpoint(
        response: Response,
        request: Request,
        resendCheckpointDto: ResendCheckpointDto,
    ): Promise<CheckpointResponse> {
        let auditChallenge: LoginChallengeSelect | null = null;
        let auditRequestMetadata: { ipAddress: string; userAgent: string } | null = null;
        let auditFailureReason = 'unexpected_error';

        try {
            const { checkpointToken } = resendCheckpointDto;
            const now = new Date();
            const challenge = await this.loginChallengeService.getChallenge(checkpointToken);

            if (!challenge) {
                auditFailureReason = 'challenge_not_found';
                throw new UnauthorizedException('Invalid or expired login challenge.');
            }

            auditChallenge = challenge;
            const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
            auditRequestMetadata = { ipAddress, userAgent };

            const latestChallenge = await this.loginChallengeService.getLatestChallengeForDevice(
                challenge.userId,
                challenge.realm as AuthRealm,
                challenge.deviceId,
            );
            const eligibility = this.loginChallengeService.getResendEligibility(
                challenge,
                now,
                latestChallenge,
            );

            if (!eligibility.eligible) {
                auditFailureReason = `ineligible_${eligibility.reason}`;
                throw new UnauthorizedException('Invalid or expired login challenge.');
            }

            const user = await this.userService.findById(challenge.userId);
            if (!user || !user.isActive) {
                auditFailureReason = 'user_not_found_or_inactive';
                throw new UnauthorizedException('User not found or inactive.');
            }

            const policy = await this.loginChallengeResendPolicyService.reserveResendAttempt({
                userId: challenge.userId,
                realm: challenge.realm,
                deviceId: challenge.deviceId,
                ipAddress,
            });

            if (!policy.allowed) {
                auditFailureReason = policy.reason
                    ? `rate_limited_${policy.reason}`
                    : 'rate_limited';
                this.setRetryAfterHeader(response, policy.retryAfterSeconds);
                throw new HttpException(
                    'Too many verification code requests.',
                    HttpStatus.TOO_MANY_REQUESTS,
                );
            }

            const geo = await this.geoIpService.lookup(ipAddress);
            let checkpointResponse: CheckpointResponse | null = null;

            await this.transactionRepository.run(async transaction => {
                auditFailureReason = 'challenge_creation_failed';
                const {
                    challenge: newChallenge,
                    checkpointToken: newCheckpointToken,
                    code,
                } = await this.loginChallengeService.createLoginChallenge(
                    challenge.userId,
                    challenge.realm as AuthRealm,
                    challenge.knownDeviceId,
                    challenge.deviceId,
                    {
                        ipAddress,
                        userAgent,
                        country: geo.country ?? undefined,
                        region: geo.region ?? undefined,
                        city: geo.city ?? undefined,
                    },
                    {
                        riskScore: challenge.riskScore,
                        riskReason: challenge.riskReason ?? undefined,
                    },
                    now,
                    transaction,
                );

                auditFailureReason = 'mail_delivery_failed';
                await this.mailTemplateService.sendVerificationCodeOrThrow(
                    user.email,
                    user.name || 'User',
                    code,
                    LOGIN_CHALLENGE_TTL_MINUTES,
                    transaction,
                );

                auditFailureReason = 'success_audit_failed';
                await this.securityEventService.recordLoginApprovalResent(
                    {
                        userId: challenge.userId,
                        realm: challenge.realm as AuthRealm,
                        ipAddress,
                        userAgent,
                        metadata: {
                            oldLoginChallengeId: challenge.id,
                            newLoginChallengeId: newChallenge.id,
                            deviceId: challenge.deviceId,
                        },
                    },
                    transaction,
                );

                checkpointResponse = {
                    checkpointRequired: true,
                    loginChallengeId: newChallenge.id,
                    checkpointToken: newCheckpointToken,
                    expiresInSeconds: LOGIN_CHALLENGE_TTL_SECONDS,
                    resendAvailableInSeconds: LOGIN_CHALLENGE_RESEND_COOLDOWN_SECONDS,
                };
            });

            if (!checkpointResponse) {
                auditFailureReason = 'checkpoint_response_missing';
                throw new InternalServerErrorException('Login checkpoint resend failed.');
            }

            this.deviceIdService.setDeviceIdCookie(response, challenge.deviceId);

            return checkpointResponse;
        } catch (error) {
            if (auditChallenge) {
                await this.securityEventService.recordLoginApprovalResendFailed({
                    userId: auditChallenge.userId,
                    realm: auditChallenge.realm as AuthRealm,
                    ipAddress: auditRequestMetadata?.ipAddress,
                    userAgent: auditRequestMetadata?.userAgent,
                    metadata: {
                        oldLoginChallengeId: auditChallenge.id,
                        deviceId: auditChallenge.deviceId,
                        failureReason: auditFailureReason,
                    },
                });
            }

            this.handleUnexpectedError(error, 'Failed to resend login challenge');
        }
    }

    async logout(request: Request, response: Response, userRole?: UserRoles): Promise<boolean> {
        let role = userRole;
        const cookies = this.getRequestCookies(request);

        if (!role) {
            const adminToken = cookies[COOKIE_NAMES.ADMIN_ACCESS_TOKEN];
            const customerToken = cookies[COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN];

            if (adminToken) {
                try {
                    const payload = this.tokenService.getTokenData(
                        adminToken,
                    ) as AccessTokenPayload;
                    role = payload.role as UserRoles;
                } catch {
                    role = UserRoles.PLATFORM_ADMIN;
                }
            } else if (customerToken) {
                try {
                    const payload = this.tokenService.getTokenData(
                        customerToken,
                    ) as AccessTokenPayload;
                    role = payload.role as UserRoles;
                } catch {
                    // ignore
                }
            }
        }

        if (!role) {
            return false;
        }

        const refreshTokenCookieName = this.authSessionService.getCookieName(
            role,
            TokenType.REFRESH,
        );
        const accessTokenCookieName = this.authSessionService.getCookieName(role, TokenType.ACCESS);
        const accessToken = cookies[accessTokenCookieName];
        const refreshToken = cookies[refreshTokenCookieName];
        const expectedRealm = deriveAuthRealmFromRole(role);
        const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
        let accessTokenPayload: AccessTokenPayload | null = null;
        let refreshTokenValid: RefreshTokenVerificationResult | null = null;

        if (accessToken) {
            try {
                accessTokenPayload = this.tokenService.getTokenData(
                    accessToken,
                ) as AccessTokenPayload;
            } catch (error) {
                if (!this.isInvalidTokenError(error)) {
                    throw error;
                }
            }
        }

        if (refreshToken) {
            try {
                refreshTokenValid = await this.tokenService.verifyRefreshToken(
                    refreshToken,
                    expectedRealm,
                );
            } catch (error) {
                if (!this.isInvalidTokenError(error)) {
                    throw error;
                }
            }
        }

        const sessionId = refreshTokenValid?.verified
            ? refreshTokenValid.sessionId
            : accessTokenPayload?.sessionId;
        const userId = refreshTokenValid?.verified
            ? refreshTokenValid.user.id
            : accessTokenPayload?.userId;

        if (!sessionId || !userId) {
            return false;
        }

        await this.sessionService.revokeCurrentSession({
            sessionId,
            userId,
            realm: expectedRealm,
            accessTokenJti: accessTokenPayload?.jti,
            ipAddress,
            userAgent,
        });

        this.authSessionService.clearTokens(response, role);

        return true;
    }

    getMe(request: Request, userRole?: UserRoles): User {
        if (request.user) {
            const { userId, email, username, role } = request.user as AccessTokenPayload;

            return {
                id: userId,
                email,
                name: username,
                role: role as UserRoles,
            };
        }

        const cookies = this.getRequestCookies(request);
        let token: string | undefined;

        if (userRole === UserRoles.PLATFORM_ADMIN || userRole === UserRoles.SUPER_ADMIN) {
            token = cookies[COOKIE_NAMES.ADMIN_ACCESS_TOKEN];
        } else if (userRole === UserRoles.USER) {
            token = cookies[COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN];
        } else {
            token =
                cookies[COOKIE_NAMES.ADMIN_ACCESS_TOKEN] ||
                cookies[COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN];
        }

        if (!token) {
            throw new UnauthorizedException('Authentication token is missing');
        }

        try {
            const {
                userId,
                email,
                username,
                role: tokenRole,
            } = this.tokenService.getTokenData(token) as AccessTokenPayload;

            return {
                id: userId,
                email,
                name: username,
                role: tokenRole as UserRoles,
            };
        } catch {
            throw new UnauthorizedException('Invalid authentication token');
        }
    }

    private getRequestCookies(request: Request): RequestCookies {
        const parsedCookies = this.parseCookies(request.headers.cookie);
        const rawCookies: unknown = request.cookies;

        if (!this.isCookieRecord(rawCookies)) {
            return parsedCookies;
        }

        return {
            ...parsedCookies,
            ...rawCookies,
        };
    }

    private isCookieRecord(value: unknown): value is RequestCookies {
        return (
            typeof value === 'object' &&
            value !== null &&
            Object.values(value).every(item => item === undefined || typeof item === 'string')
        );
    }

    private parseCookies(cookieHeader?: string): RequestCookies {
        if (!cookieHeader) return {};

        return cookieHeader.split(';').reduce<RequestCookies>((cookies, item) => {
            const [rawKey, ...rawValueParts] = item.trim().split('=');

            if (!rawKey) {
                return cookies;
            }

            cookies[decodeURIComponent(rawKey)] = decodeURIComponent(rawValueParts.join('='));

            return cookies;
        }, {});
    }

    async refreshAccessToken(
        response: Response,
        request: Request,
        tokenType: string,
    ): Promise<boolean> {
        const cookies = this.getRequestCookies(request);
        const cookiesKeys = Object.keys(cookies);
        const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
        const { deviceId } = this.deviceIdService.getOrCreateDeviceId(request);

        let expectedRealm: AuthRealm | undefined;
        if (tokenType === (COOKIE_NAMES.ADMIN_REFRESH_TOKEN as string)) {
            expectedRealm = 'admin';
        } else if (tokenType === (COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN as string)) {
            expectedRealm = 'customer';
        }

        if (!cookiesKeys.includes(tokenType)) {
            return false;
        }

        const refreshToken = cookies[tokenType];

        if (!refreshToken) {
            return false;
        }

        let verificationResult: RefreshTokenVerificationResult;
        try {
            verificationResult = await this.tokenService.verifyRefreshToken(
                refreshToken,
                expectedRealm,
            );
        } catch (error) {
            let userId: string | undefined;
            try {
                const payload = this.tokenService.getTokenData(refreshToken);
                if (payload && 'userId' in payload) {
                    userId = payload.userId;
                }
            } catch {
                void 0;
            }

            await this.securityEventService.recordRefreshFailed({
                userId,
                realm: expectedRealm ?? 'customer',
                ipAddress,
                userAgent,
                metadata: {
                    failureReason:
                        error instanceof Error
                            ? error.message
                            : 'Token verification threw an error',
                    deviceId,
                },
            });
            return false;
        }

        if (!verificationResult.verified) {
            let userId: string | undefined;
            try {
                const payload = this.tokenService.getTokenData(refreshToken);
                if (payload && 'userId' in payload) {
                    userId = payload.userId;
                }
            } catch {
                void 0;
            }

            await this.securityEventService.recordRefreshFailed({
                userId,
                realm: expectedRealm ?? 'customer',
                ipAddress,
                userAgent,
                metadata: {
                    failureReason:
                        'Verification failed: token or session invalid/mismatched/revoked',
                    deviceId,
                },
            });
            return false;
        }

        const { id, name, email, role } = verificationResult.user;
        const { sessionId, tokenId, isGrace } = verificationResult;

        if (id && name && email && role && sessionId) {
            const userJWTPayload = {
                id,
                name,
                email,
                role,
                ipAddress,
                sessionId,
                userAgent,
            };

            const rotationEnabled =
                this.configService.get<boolean>('REFRESH_ROTATION_ENABLED') ?? false;

            if (isGrace && !rotationEnabled) {
                await this.securityEventService.recordRefreshFailed({
                    userId: id,
                    realm: expectedRealm ?? 'customer',
                    ipAddress,
                    userAgent,
                    metadata: {
                        failureReason: 'Grace refresh attempted but rotation is disabled',
                        deviceId,
                    },
                });
                return false;
            }

            const session = await this.sessionService.getSession(sessionId);
            if (!session) {
                await this.securityEventService.recordRefreshFailed({
                    userId: id,
                    realm: expectedRealm ?? 'customer',
                    ipAddress,
                    userAgent,
                    metadata: {
                        failureReason: 'Session not found in DB',
                        deviceId,
                    },
                });
                return false;
            }

            const realm = session.realm;
            const deviceBindingRequired =
                realm === 'admin'
                    ? (this.configService.get<boolean>('DEVICE_BINDING_REQUIRED_ADMIN') ?? false)
                    : (this.configService.get<boolean>('DEVICE_BINDING_REQUIRED_CUSTOMER') ??
                      false);

            const requestDeviceId = this.deviceIdService.readDeviceId(request);

            if (deviceBindingRequired) {
                if (!requestDeviceId || requestDeviceId !== session.deviceId) {
                    await this.securityEventService.recordRefreshFailed({
                        userId: id,
                        realm: expectedRealm ?? 'customer',
                        ipAddress,
                        userAgent,
                        metadata: {
                            failureReason: !requestDeviceId
                                ? 'Device binding mismatch: deviceId cookie missing'
                                : `Device binding mismatch: cookie deviceId (${requestDeviceId}) does not match session deviceId (${session.deviceId})`,
                            deviceId: requestDeviceId ?? undefined,
                        },
                    });
                    throw new UnauthorizedException('Device binding validation failed.');
                }
            } else {
                if (!requestDeviceId) {
                    this.logger.warn(
                        `[Device Binding Shadow Mode] Missing deviceId cookie for userId: ${id}, sessionId: ${sessionId}`,
                    );
                } else if (requestDeviceId !== session.deviceId) {
                    this.logger.warn(
                        `[Device Binding Shadow Mode] deviceId mismatch for userId: ${id}, sessionId: ${sessionId}. Cookie deviceId: ${requestDeviceId}, Session deviceId: ${session.deviceId}`,
                    );
                }
            }

            // Throttled updates
            const throttleWindow = 60; // 60 seconds
            await this.sessionService.touchLastSeenAt(sessionId, throttleWindow);
            if (session.knownDeviceId) {
                await this.knownDeviceService.touchKnownDevice(
                    session.knownDeviceId,
                    {
                        lastIpAddress: ipAddress,
                        lastUserAgent: userAgent,
                    },
                    throttleWindow,
                );
            }

            this.deviceIdService.setDeviceIdCookie(response, deviceId);

            if (isGrace) {
                const replacementRefreshToken =
                    await this.tokenService.acceptReplacedTokenInGraceById(tokenId);
                if (!replacementRefreshToken) {
                    await this.securityEventService.recordRefreshFailed({
                        userId: id,
                        realm: expectedRealm ?? 'customer',
                        ipAddress,
                        userAgent,
                        metadata: {
                            failureReason: 'Grace refresh replacement token unavailable',
                            deviceId,
                        },
                    });
                    return false;
                }

                const accessToken = this.tokenService.createAccessToken(userJWTPayload);
                this.authSessionService.applyTokens(response, {
                    accessToken,
                    refreshToken: replacementRefreshToken,
                    role: userJWTPayload.role,
                });
            } else if (rotationEnabled) {
                // Current token + rotation enabled: rotate refresh token
                const newRawRefreshToken = await this.tokenService.rotateRefreshTokenById(
                    tokenId,
                    session,
                );
                if (!newRawRefreshToken) {
                    await this.securityEventService.recordRefreshFailed({
                        userId: id,
                        realm: expectedRealm ?? 'customer',
                        ipAddress,
                        userAgent,
                        metadata: {
                            failureReason: 'Token rotation failed',
                            deviceId,
                        },
                    });
                    return false;
                }
                const accessToken = this.tokenService.createAccessToken(userJWTPayload);
                this.authSessionService.applyTokens(response, {
                    accessToken,
                    refreshToken: newRawRefreshToken,
                    role: userJWTPayload.role,
                });
            } else {
                // Current token + rotation disabled: do not rotate
                await this.authSessionService.issueTokens(response, userJWTPayload, false);
            }

            return true;
        }

        return false;
    }

    private async verifyPassword(password: string, passwordHash: string): Promise<boolean> {
        try {
            return await Argon2HashUtil.compare(password, passwordHash);
        } catch {
            return false;
        }
    }

    private handleUnexpectedError(error: unknown, message: string): never {
        if (error instanceof HttpException) {
            throw error;
        }

        throw new InternalServerErrorException(message);
    }

    private setRetryAfterHeader(response: Response, retryAfterSeconds: number): void {
        const responseWithHeader = response as Response & {
            setHeader?: (name: string, value: string) => void;
        };

        if (typeof responseWithHeader.setHeader === 'function') {
            responseWithHeader.setHeader('Retry-After', String(retryAfterSeconds));
        }
    }

    private isInvalidTokenError(error: unknown): boolean {
        if (error instanceof UnauthorizedException) {
            return true;
        }

        if (error instanceof Error) {
            return ['JsonWebTokenError', 'NotBeforeError', 'TokenExpiredError'].includes(
                error.name,
            );
        }

        return false;
    }

    private async evaluateLoginRisk(
        user: UserModel,
        realm: AuthRealm,
        deviceId: string,
        currentMetadata: {
            ipAddress: string;
            userAgent: string;
            country?: string;
            region?: string;
            city?: string;
        },
        transaction?: RepositoryTransaction,
    ): Promise<RiskDecision> {
        const userId = user.id;
        const isEmailVerified = user.emailVerifiedAt !== null;

        const events = await this.securityEventRepository.findByUserId(userId, transaction);
        const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        const eventsLast24h = events.filter(e => e.createdAt.getTime() >= dayAgo.getTime());

        const failedLoginAttempts24h = eventsLast24h.filter(
            e => e.eventType === SecurityEventType.LOGIN_FAILED,
        ).length;

        const failedRefreshAttempts24h = eventsLast24h.filter(
            e => e.eventType === SecurityEventType.REFRESH_FAILED,
        ).length;

        const isPasswordRecentlyReset = eventsLast24h.some(
            e =>
                e.eventType === SecurityEventType.PASSWORD_CHANGED ||
                e.eventType === SecurityEventType.PASSWORD_RESET_COMPLETED,
        );

        const activeDevices = await this.knownDeviceService.listKnownDevices(
            userId,
            realm,
            transaction,
        );

        const mappedActiveDevices = activeDevices.map(d => ({
            deviceId: d.deviceId,
            trustedAt: d.trustedAt,
            trustExpiresAt: d.trustExpiresAt,
            lastIpAddress: d.lastIpAddress,
            lastCountry: d.lastCountry,
            lastRegion: d.lastRegion,
            lastCity: d.lastCity,
            lastUserAgent: d.lastUserAgent,
        }));

        const lastLoginEvent = events
            .filter(e => e.eventType === SecurityEventType.LOGIN_SUCCESS)
            .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())[0];

        const history: UserSecurityHistory = {
            activeDevices: mappedActiveDevices,
            lastLoginEvent: lastLoginEvent
                ? {
                      timestamp: lastLoginEvent.createdAt,
                      ipAddress: lastLoginEvent.ipAddress,
                      country: lastLoginEvent.country,
                      region: lastLoginEvent.region,
                      city: lastLoginEvent.city,
                  }
                : null,
        };

        const context: RiskEvaluationContext = {
            userId,
            realm,
            deviceId,
            currentMetadata,
            isEmailVerified,
            isPasswordRecentlyReset,
            deviceBindingFailed: false,
            failedLoginAttempts24h,
            failedRefreshAttempts24h,
            isRegistrationBootstrap: false,
        };

        return this.riskPolicyService.evaluateRisk(context, history);
    }
}
