import {
    BadRequestException,
    ConflictException,
    HttpException,
    Injectable,
    InternalServerErrorException,
    UnauthorizedException,
} from '@nestjs/common';
import { AuthService } from '@/module-auth/services/auth.service';
import type { Request, Response } from 'express';
import { User, UserJwtPayload, CheckpointResponse } from '@/module-user/interfaces/user.interfaces';
import { RegisterDto } from '@/module-auth/dto/register-user.dto';
import { LoginUserDto } from '@/module-auth/dto/login-user.dto';
import { VerifyCheckpointDto } from '@/module-auth/dto/verify-checkpoint.dto';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';
import { UserRoles, COOKIE_NAMES, deriveAuthRealmFromRole } from '@/module-auth/enums/auth.enums';
import { UserRepository } from '@/module-user/repository/user.repository';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { UserUniquenessService } from '@/module-user/services/user-uniqueness.service';
import { UserSelect } from '@/module-user/schemas/users.schema';
import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { SessionService } from '@/module-auth-token/services/session.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import { TransactionRepository } from '@/module-drizzle/repository/transaction.repository';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { EmailVerificationService } from '@/module-auth-token/services/email-verification.service';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

@Injectable()
export class CustomerAuthService {
    private readonly customerRoles = [UserRoles.USER];

    public constructor(
        private readonly authService: AuthService,
        private readonly userRepository: UserRepository,
        private readonly userPasswordService: UserPasswordService,
        private readonly userUniquenessService: UserUniquenessService,
        private readonly deviceIdService: DeviceIdService,
        private readonly knownDeviceService: KnownDeviceService,
        private readonly sessionService: SessionService,
        private readonly securityEventService: SecurityEventService,
        private readonly geoIpService: GeoIpService,
        private readonly transactionRepository: TransactionRepository,
        private readonly authSessionService: AuthSessionService,
        private readonly emailVerificationService: EmailVerificationService,
        private readonly identityIntegrationService: IdentityIntegrationService,
    ) {}

    async register(response: Response, request: Request, registerDto: RegisterDto): Promise<User> {
        let createdUser: UserSelect | undefined;
        let deviceId: string | undefined;
        let tokens:
            | { accessToken: string; refreshToken: string | null; role: UserRoles }
            | undefined;
        let createdTenantId: string | undefined;

        const { ipAddress, userAgent } = this.authSessionService.getUserMetaData(request);
        const geo = await this.geoIpService.lookup(ipAddress);

        try {
            await this.transactionRepository.run(async transaction => {
                await this.userUniquenessService.ensureUnique(
                    registerDto.email,
                    registerDto.name,
                    undefined,
                    transaction,
                );

                const hashedPassword = await this.userPasswordService.hashPassword(
                    registerDto.password,
                );

                const result = await this.userRepository.createWithTenant(
                    {
                        name: registerDto.name,
                        email: registerDto.email,
                        password: hashedPassword,
                        role: UserRoles.USER,
                    },
                    `${registerDto.name}'s Space`,
                    transaction,
                );

                createdUser = result.user;
                createdTenantId = result.tenant.id;

                const userId = createdUser.id;
                const verificationToken =
                    await this.emailVerificationService.createVerificationToken(
                        userId,
                        transaction,
                    );
                await this.emailVerificationService.sendVerificationEmail(
                    userId,
                    createdUser.email,
                    createdUser.name,
                    verificationToken,
                    transaction,
                );
                const realm = deriveAuthRealmFromRole(createdUser.role as UserRoles);

                const deviceResult = this.deviceIdService.getOrCreateDeviceId(request);
                deviceId = deviceResult.deviceId;

                const knownDeviceMetadata = {
                    lastIpAddress: ipAddress,
                    lastCountry: geo.country,
                    lastRegion: geo.region,
                    lastCity: geo.city,
                    lastUserAgent: userAgent,
                };
                const knownDevice = await this.knownDeviceService.findOrCreateKnownDevice(
                    userId,
                    realm,
                    deviceId,
                    knownDeviceMetadata,
                    transaction,
                );

                const sessionMetadata = {
                    ipAddress,
                    lastCountry: geo.country,
                    lastRegion: geo.region,
                    lastCity: geo.city,
                    userAgent,
                };
                const session = await this.sessionService.createSession(
                    userId,
                    realm,
                    knownDevice.id,
                    deviceId,
                    sessionMetadata,
                    transaction,
                );
                const sessionId = session.id;

                const userJWTPayload: UserJwtPayload = {
                    id: createdUser.id,
                    name: createdUser.name,
                    email: createdUser.email,
                    role: createdUser.role as UserRoles,
                    sessionId,
                    ipAddress,
                    userAgent,
                };

                const issued = await this.authSessionService.issueTokensDeferred(
                    userJWTPayload,
                    true,
                    transaction,
                );

                tokens = {
                    accessToken: issued.accessToken,
                    refreshToken: issued.refreshToken,
                    role: createdUser.role as UserRoles,
                };

                await this.securityEventService.recordRegistrationSuccess(
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

                await this.identityIntegrationService.recordUserRegistered(
                    createdUser,
                    createdTenantId,
                    transaction,
                );
            });
        } catch (error) {
            this.handleUnexpectedDatabaseError(
                error,
                'Failed to create generic user registration.',
            );
        }

        if (!createdUser || !createdTenantId || !deviceId || !tokens) {
            throw new InternalServerErrorException('Registration failed during session issuance.');
        }

        // Apply cookies and return success ONLY after commit
        this.deviceIdService.setDeviceIdCookie(response, deviceId);
        this.authSessionService.applyTokens(response, tokens);

        return {
            id: createdUser.id,
            name: createdUser.name,
            email: createdUser.email,
            role: createdUser.role as UserRoles,
        };
    }

    private handleUnexpectedDatabaseError(error: unknown, message: string): never {
        if (error instanceof HttpException) {
            throw error;
        }

        if (this.isUniqueViolation(error)) {
            throw new ConflictException('User with this email or user name already exists.');
        }

        throw new InternalServerErrorException(message);
    }

    private isUniqueViolation(error: unknown): boolean {
        return Boolean(
            typeof error === 'object' &&
            error !== null &&
            'code' in error &&
            (error as { code?: string }).code === '23505',
        );
    }

    async login(
        response: Response,
        request: Request,
        loginUserDto: LoginUserDto,
    ): Promise<User | CheckpointResponse> {
        return this.authService.login(response, request, loginUserDto, this.customerRoles);
    }

    async verifyLoginCheckpoint(
        response: Response,
        request: Request,
        verifyCheckpointDto: VerifyCheckpointDto,
    ): Promise<User> {
        return this.authService.verifyLoginCheckpoint(response, request, verifyCheckpointDto);
    }

    async resendLoginCheckpoint(
        response: Response,
        request: Request,
        resendCheckpointDto: ResendCheckpointDto,
    ): Promise<CheckpointResponse> {
        return this.authService.resendLoginCheckpoint(response, request, resendCheckpointDto);
    }

    async logout(request: Request, response: Response, role?: UserRoles): Promise<void> {
        this.assertCustomerRole(role);

        await this.authService.logout(request, response, role);
    }

    async refresh(response: Response, request: Request, role?: UserRoles): Promise<boolean> {
        this.assertCustomerRole(role);

        const cookieName = COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN;

        if (!request.cookies?.[cookieName]) {
            throw new UnauthorizedException('Refresh token is missing');
        }

        try {
            const success = await this.authService.refreshAccessToken(
                response,
                request,
                cookieName,
            );
            if (!success) {
                throw new UnauthorizedException('Refresh token is invalid or revoked');
            }
            return true;
        } catch (error) {
            if (error instanceof HttpException) {
                throw error;
            }

            throw new InternalServerErrorException(`Failed to refresh token`);
        }
    }

    getMe(request: Request, role?: UserRoles): User {
        this.assertCustomerRole(role);

        return this.authService.getMe(request, role);
    }

    private assertCustomerRole(role?: UserRoles): void {
        if (role && !this.customerRoles.includes(role)) {
            throw new BadRequestException('Customer auth supports only the user role.');
        }
    }

    async verifyEmail(token: string): Promise<boolean> {
        return this.emailVerificationService.verifyEmail(
            token,
            undefined,
            (user, transaction) =>
                this.identityIntegrationService.recordEmailVerified(user, transaction),
        );
    }

    async resendVerification(userId: string): Promise<void> {
        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new BadRequestException('User not found.');
        }
        await this.emailVerificationService.resendVerification(userId, user.email, user.name);
    }
}
