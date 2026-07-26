import {
    Body,
    Controller,
    Delete,
    Get,
    HttpCode,
    HttpStatus,
    Param,
    ParseUUIDPipe,
    Post,
    Req,
    Res,
} from '@nestjs/common';
import { ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import type { Request, Response } from 'express';
import { EmailFlowRateLimit } from '@/common/rate-limiting/decorators/email-flow-rate-limit.decorator';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { CurrentUser } from '@/module-auth/decorators/current-user.decorator';
import { Public } from '@/module-auth/decorators/public.decorator';
import { ReauthConfirmationToken } from '@/module-auth/decorators/reauth-confirmation-token.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { ReauthDto } from '@/module-auth-token/dto/re-auth.dto';
import type { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import {
    ChangeEmailConfirmDto,
    ChangeEmailRequestDto,
    ChangePasswordDto,
    ResetPasswordConfirmDto,
    ResetPasswordRequestDto,
} from '@/module-account/dto/account-settings.dto';
import { AccountSettingsService } from '@/module-account/services/account-settings.service';

const IDENTITY_ROLES = [
    UserRoles.USER,
    UserRoles.PLATFORM_ADMIN,
    UserRoles.SUPER_ADMIN,
];

@ApiTags('Account Security')
@Controller()
export class AccountSecurityController {
    constructor(
        private readonly accountSettingsService: AccountSettingsService,
        private readonly authSessionService: AuthSessionService,
        private readonly reauthConfirmationService: ReauthConfirmationService,
        private readonly knownDeviceService: KnownDeviceService,
    ) {}

    @Post('auth/password/forgot')
    @Public()
    @EmailFlowRateLimit({ flow: 'password_reset', recipientBodyField: 'email' })
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Request a password reset link' })
    async forgotPassword(@Body() dto: ResetPasswordRequestDto) {
        await this.accountSettingsService.resetPasswordRequest(dto.email);
        return { success: true };
    }

    @Get('account/devices')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'List active known devices for the current account' })
    getKnownDevices(@CurrentUser() user: AccessTokenPayload) {
        return this.knownDeviceService.listKnownDevices(user.userId, user.realm);
    }

    @Delete('account/devices/:id')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Revoke a known device and every session created on it' })
    async revokeKnownDevice(
        @CurrentUser() user: AccessTokenPayload,
        @Param('id', ParseUUIDPipe) knownDeviceId: string,
    ) {
        await this.knownDeviceService.revokeKnownDevice(user.userId, user.realm, knownDeviceId);
        return { success: true };
    }

    @Post('auth/password/reset')
    @Public()
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Reset a password with a one-time token' })
    async resetPassword(
        @Body() dto: ResetPasswordConfirmDto,
        @Req() request: Request,
        @Res({ passthrough: true }) response: Response,
    ) {
        const metadata = this.authSessionService.getUserMetaData(request);
        await this.accountSettingsService.resetPasswordConfirm(dto.token, dto, metadata);
        this.clearIdentityCookies(response);
        return { success: true };
    }

    @Get('account/sessions')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'List active sessions for the current account' })
    getSessions(@CurrentUser() user: AccessTokenPayload) {
        return this.accountSettingsService.getActiveSessions(
            user.userId,
            user.realm,
            user.sessionId,
        );
    }

    @Delete('account/sessions')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Revoke every active session for the current account' })
    async revokeAllSessions(
        @CurrentUser() user: AccessTokenPayload,
        @Res({ passthrough: true }) response: Response,
    ) {
        await this.accountSettingsService.revokeAllSessions(
            user.userId,
            user.realm,
            user.sessionId,
        );
        this.clearIdentityCookies(response);
        return { success: true };
    }

    @Delete('account/sessions/others')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Revoke every session except the current session' })
    async revokeOtherSessions(
        @CurrentUser() user: AccessTokenPayload,
        @ReauthConfirmationToken() reauthConfirmationToken?: string,
    ) {
        await this.accountSettingsService.revokeOtherSessions(
            user.userId,
            user.realm,
            user.sessionId,
            reauthConfirmationToken,
        );
        return { success: true };
    }

    @Delete('account/sessions/:id')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Revoke one session owned by the current account' })
    async revokeSession(
        @CurrentUser() user: AccessTokenPayload,
        @Param('id', ParseUUIDPipe) sessionId: string,
        @Res({ passthrough: true }) response: Response,
    ) {
        await this.accountSettingsService.revokeSession(user.userId, user.realm, sessionId);
        if (sessionId === user.sessionId) {
            this.clearIdentityCookies(response);
        }
        return { success: true };
    }

    @Post('account/password/change')
    @Authorisation(...IDENTITY_ROLES)
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Change the password for the current account' })
    async changePassword(
        @CurrentUser() user: AccessTokenPayload,
        @Body() dto: ChangePasswordDto,
        @Req() request: Request,
        @Res({ passthrough: true }) response: Response,
        @ReauthConfirmationToken() reauthConfirmationToken?: string,
    ) {
        const metadata = this.authSessionService.getUserMetaData(request);
        await this.accountSettingsService.changePassword(
            user.userId,
            user.realm,
            user.sessionId,
            dto,
            reauthConfirmationToken,
            metadata,
        );
        this.clearIdentityCookies(response);
        return { success: true };
    }

    @Post('account/email/change/request')
    @Authorisation(...IDENTITY_ROLES)
    @EmailFlowRateLimit({ flow: 'email_change', recipientBodyField: 'newEmail' })
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Request a change to the account email address' })
    async requestEmailChange(
        @CurrentUser() user: AccessTokenPayload,
        @Body() dto: ChangeEmailRequestDto,
        @Req() request: Request,
        @ReauthConfirmationToken() reauthConfirmationToken?: string,
    ) {
        const metadata = this.authSessionService.getUserMetaData(request);
        await this.accountSettingsService.changeEmailRequest(
            user.userId,
            user.realm,
            user.sessionId,
            dto.newEmail,
            reauthConfirmationToken,
            metadata,
        );
        return { success: true };
    }

    @Post('account/email/change/confirm')
    @Authorisation(...IDENTITY_ROLES)
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Confirm and apply an account email change' })
    async confirmEmailChange(
        @CurrentUser() user: AccessTokenPayload,
        @Body() dto: ChangeEmailConfirmDto,
        @Res({ passthrough: true }) response: Response,
    ) {
        await this.accountSettingsService.changeEmailConfirm(user.userId, dto.code);
        this.clearIdentityCookies(response);
        return { success: true };
    }

    @Post('account/deactivate')
    @Authorisation(...IDENTITY_ROLES)
    @HttpCode(HttpStatus.OK)
    @ApiOperation({ summary: 'Deactivate the current account' })
    async deactivateAccount(
        @CurrentUser() user: AccessTokenPayload,
        @Res({ passthrough: true }) response: Response,
        @ReauthConfirmationToken() reauthConfirmationToken?: string,
    ) {
        await this.accountSettingsService.deactivateAccount(
            user.userId,
            user.realm,
            user.sessionId,
            reauthConfirmationToken,
        );
        this.clearIdentityCookies(response);
        return { success: true };
    }

    @Delete('account')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Schedule deletion of the current account' })
    async scheduleAccountDeletion(
        @CurrentUser() user: AccessTokenPayload,
        @ReauthConfirmationToken() reauthConfirmationToken?: string,
    ) {
        await this.accountSettingsService.scheduleAccountDeletion(
            user.userId,
            user.realm,
            user.sessionId,
            reauthConfirmationToken,
        );
        return { success: true };
    }

    @Post('account/re-auth')
    @Authorisation(...IDENTITY_ROLES)
    @ApiOperation({ summary: 'Create a one-time re-authentication confirmation token' })
    @ApiResponse({ status: 201, description: 'Confirmation token created.' })
    async createReauthConfirmation(
        @CurrentUser() user: AccessTokenPayload,
        @Body() dto: ReauthDto,
        @Req() request: Request,
    ) {
        const metadata = this.authSessionService.getUserMetaData(request);
        const token = await this.reauthConfirmationService.createReauthConfirmation(
            user.userId,
            user.realm,
            user.sessionId,
            dto.actionScope,
            dto.password,
            metadata,
        );
        return {
            confirmationToken: token,
            expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
        };
    }

    private clearIdentityCookies(response: Response): void {
        this.authSessionService.clearTokens(response, UserRoles.USER);
        this.authSessionService.clearTokens(response, UserRoles.PLATFORM_ADMIN);
    }
}
