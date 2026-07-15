import { INestApplication, Module } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import { AccountSecurityController } from '@/module-account/controllers/account-security.controller';
import { AccountSettingsService } from '@/module-account/services/account-settings.service';
import { JwtAuthGuard } from '@/module-auth/guards/jwt-auth.guard';
import { RolesGuard } from '@/module-auth/guards/roles.guard';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';

const accountSettingsServiceMock = {
    getActiveSessions: jest.fn(),
    revokeSession: jest.fn(),
    revokeOtherSessions: jest.fn(),
    revokeAllSessions: jest.fn(),
    changePassword: jest.fn(),
    resetPasswordRequest: jest.fn(),
    resetPasswordConfirm: jest.fn(),
    changeEmailRequest: jest.fn(),
    changeEmailConfirm: jest.fn(),
    deactivateAccount: jest.fn(),
    scheduleAccountDeletion: jest.fn(),
};

const authSessionServiceMock = {
    clearTokens: jest.fn(),
    getUserMetaData: jest.fn().mockReturnValue({
        ipAddress: '127.0.0.1',
        userAgent: 'account-security-test',
    }),
};

const reauthConfirmationServiceMock = {
    createReauthConfirmation: jest.fn(),
};

const knownDeviceServiceMock = {
    listKnownDevices: jest.fn(),
    revokeKnownDevice: jest.fn(),
};

@Module({
    controllers: [AccountSecurityController],
    providers: [
        { provide: AccountSettingsService, useValue: accountSettingsServiceMock },
        { provide: AuthSessionService, useValue: authSessionServiceMock },
        { provide: ReauthConfirmationService, useValue: reauthConfirmationServiceMock },
        { provide: KnownDeviceService, useValue: knownDeviceServiceMock },
        RolesGuard,
    ],
})
class AccountSecurityTestModule {}

describe('AccountSecurityController routes', () => {
    let app: INestApplication;
    let originalJwtCanActivate: typeof JwtAuthGuard.prototype.canActivate;
    const currentSessionId = '11111111-1111-4111-8111-111111111111';
    const otherSessionId = '22222222-2222-4222-8222-222222222222';

    beforeAll(async () => {
        // eslint-disable-next-line @typescript-eslint/unbound-method
        originalJwtCanActivate = JwtAuthGuard.prototype.canActivate;
        JwtAuthGuard.prototype.canActivate = function (this: void, context): Promise<boolean> {
            const req = context.switchToHttp().getRequest<{
                headers: Record<string, string>;
                user: {
                    userId: string;
                    role: string;
                    realm: 'customer' | 'admin';
                    sessionId: string;
                };
            }>();
            const role = req.headers['x-test-role'] ?? UserRoles.USER;
            req.user = {
                userId: req.headers['x-test-user-id'] ?? 'user-1',
                role,
                realm:
                    role === (UserRoles.PLATFORM_ADMIN as string) ||
                    role === (UserRoles.SUPER_ADMIN as string)
                        ? 'admin'
                        : 'customer',
                sessionId: req.headers['x-test-session-id'] ?? currentSessionId,
            };
            return Promise.resolve(true);
        };

        const moduleRef = await Test.createTestingModule({
            imports: [AccountSecurityTestModule],
        }).compile();

        app = moduleRef.createNestApplication();
        await app.init();
    });

    afterAll(async () => {
        await app.close();
        JwtAuthGuard.prototype.canActivate = originalJwtCanActivate;
    });

    beforeEach(() => {
        jest.clearAllMocks();
        authSessionServiceMock.getUserMetaData.mockReturnValue({
            ipAddress: '127.0.0.1',
            userAgent: 'account-security-test',
        });
    });

    it('exposes enumeration-safe POST /auth/password/forgot', async () => {
        accountSettingsServiceMock.resetPasswordRequest.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .post('/auth/password/forgot')
            .send({ email: 'user@example.com' });

        expect(response.status).toBe(200);
        expect(response.body).toEqual({ success: true });
        expect(accountSettingsServiceMock.resetPasswordRequest).toHaveBeenCalledWith(
            'user@example.com',
        );
    });

    it('resets a password and clears identity cookies', async () => {
        accountSettingsServiceMock.resetPasswordConfirm.mockResolvedValue(undefined);
        const dto = { token: 'selector.verifier', newPassword: 'new-password-123' };

        const response = await request(app.getHttpServer() as never)
            .post('/auth/password/reset')
            .send(dto);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.resetPasswordConfirm).toHaveBeenCalledWith(
            dto.token,
            dto,
            { ipAddress: '127.0.0.1', userAgent: 'account-security-test' },
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalledWith(
            expect.any(Object),
            UserRoles.USER,
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalledWith(
            expect.any(Object),
            UserRoles.PLATFORM_ADMIN,
        );
    });

    it('lists sessions through the neutral user realm', async () => {
        accountSettingsServiceMock.getActiveSessions.mockResolvedValue([
            { id: currentSessionId, currentSession: true },
        ]);

        const response = await request(app.getHttpServer() as never)
            .get('/account/sessions')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.getActiveSessions).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
        );
    });

    it('lists known devices through the neutral user realm', async () => {
        knownDeviceServiceMock.listKnownDevices.mockResolvedValue([{ id: otherSessionId }]);

        const response = await request(app.getHttpServer() as never)
            .get('/account/devices')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(knownDeviceServiceMock.listKnownDevices).toHaveBeenCalledWith('user-1', 'customer');
    });

    it('revokes a known device together with its sessions', async () => {
        knownDeviceServiceMock.revokeKnownDevice.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .delete(`/account/devices/${otherSessionId}`)
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(knownDeviceServiceMock.revokeKnownDevice).toHaveBeenCalledWith(
            'user-1',
            'customer',
            otherSessionId,
        );
    });

    it('revokes all sessions and clears cookies', async () => {
        accountSettingsServiceMock.revokeAllSessions.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .delete('/account/sessions')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.revokeAllSessions).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalled();
    });

    it('revokes every session except the current session', async () => {
        accountSettingsServiceMock.revokeOtherSessions.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .delete('/account/sessions/others')
            .set('x-test-role', UserRoles.USER)
            .set('x-reauth-confirmation', 'reauth-token');

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.revokeOtherSessions).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            'reauth-token',
        );
    });

    it('revokes a specific session without clearing another active session cookie', async () => {
        accountSettingsServiceMock.revokeSession.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .delete(`/account/sessions/${otherSessionId}`)
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.revokeSession).toHaveBeenCalledWith(
            'user-1',
            'customer',
            otherSessionId,
        );
        expect(authSessionServiceMock.clearTokens).not.toHaveBeenCalled();
    });

    it('changes the password and clears identity cookies', async () => {
        accountSettingsServiceMock.changePassword.mockResolvedValue(undefined);
        const dto = { oldPassword: 'old-password', newPassword: 'new-password-123' };

        const response = await request(app.getHttpServer() as never)
            .post('/account/password/change')
            .set('x-test-role', UserRoles.USER)
            .send(dto);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.changePassword).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            dto,
            undefined,
            { ipAddress: '127.0.0.1', userAgent: 'account-security-test' },
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalled();
    });

    it('requests and confirms an email change', async () => {
        accountSettingsServiceMock.changeEmailRequest.mockResolvedValue(undefined);
        accountSettingsServiceMock.changeEmailConfirm.mockResolvedValue(undefined);

        const requestResponse = await request(app.getHttpServer() as never)
            .post('/account/email/change/request')
            .set('x-test-role', UserRoles.USER)
            .send({ newEmail: 'new@example.com' });
        const confirmResponse = await request(app.getHttpServer() as never)
            .post('/account/email/change/confirm')
            .set('x-test-role', UserRoles.USER)
            .send({ code: '123456' });

        expect(requestResponse.status).toBe(200);
        expect(confirmResponse.status).toBe(200);
        expect(accountSettingsServiceMock.changeEmailRequest).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            'new@example.com',
            undefined,
            { ipAddress: '127.0.0.1', userAgent: 'account-security-test' },
        );
        expect(accountSettingsServiceMock.changeEmailConfirm).toHaveBeenCalledWith(
            'user-1',
            '123456',
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalled();
    });

    it('deactivates the current account and clears identity cookies', async () => {
        accountSettingsServiceMock.deactivateAccount.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .post('/account/deactivate')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.deactivateAccount).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            undefined,
        );
        expect(authSessionServiceMock.clearTokens).toHaveBeenCalled();
    });

    it('schedules deletion through DELETE /account', async () => {
        accountSettingsServiceMock.scheduleAccountDeletion.mockResolvedValue(undefined);

        const response = await request(app.getHttpServer() as never)
            .delete('/account')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.scheduleAccountDeletion).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            undefined,
        );
    });

    it('creates a neutral re-authentication token', async () => {
        reauthConfirmationServiceMock.createReauthConfirmation.mockResolvedValue('confirmation');

        const response = await request(app.getHttpServer() as never)
            .post('/account/re-auth')
            .set('x-test-role', UserRoles.USER)
            .send({ actionScope: 'password_change', password: 'current-password' });

        expect(response.status).toBe(201);
        expect((response.body as { confirmationToken: string }).confirmationToken).toBe(
            'confirmation',
        );
        expect(reauthConfirmationServiceMock.createReauthConfirmation).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
            'password_change',
            'current-password',
            { ipAddress: '127.0.0.1', userAgent: 'account-security-test' },
        );
    });

    it('keeps legacy customer roles compatible during the migration', async () => {
        accountSettingsServiceMock.getActiveSessions.mockResolvedValue([]);

        const response = await request(app.getHttpServer() as never)
            .get('/account/sessions')
            .set('x-test-role', UserRoles.USER);

        expect(response.status).toBe(200);
        expect(accountSettingsServiceMock.getActiveSessions).toHaveBeenCalledWith(
            'user-1',
            'customer',
            currentSessionId,
        );
    });
});
