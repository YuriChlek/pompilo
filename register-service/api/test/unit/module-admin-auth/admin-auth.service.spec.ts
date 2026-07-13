import { InternalServerErrorException, UnauthorizedException } from '@nestjs/common';
import type { Request, Response } from 'express';
import { AdminAuthService } from '@/module-admin-auth/services/admin-auth.service';
import { AuthService } from '@/module-auth/services/auth.service';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import { LoginAdminDto } from '@/module-admin-auth/dto/login-admin.dto';

describe('AdminAuthService', () => {
    let service: AdminAuthService;
    let authService: {
        login: jest.MockedFunction<AuthService['login']>;
        logout: jest.MockedFunction<AuthService['logout']>;
        refreshAccessToken: jest.MockedFunction<AuthService['refreshAccessToken']>;
        getMe: jest.MockedFunction<AuthService['getMe']>;
    };
    const request = {} as Request;
    const response = {} as Response;

    beforeEach(() => {
        authService = {
            login: jest.fn(),
            logout: jest.fn(),
            refreshAccessToken: jest.fn().mockResolvedValue(true),
            getMe: jest.fn().mockReturnValue({ id: 'admin' }),
        };

        service = new AdminAuthService(authService as unknown as AuthService);
    });

    it('logs admin in by delegating to AuthService', async () => {
        const dto: LoginAdminDto = {
            login: 'admin@example.com',
            password: 'Password1',
            role: UserRoles.PLATFORM_ADMIN,
        };

        await service.login(response, request, dto);

        expect(authService.login).toHaveBeenCalledWith(response, request, dto, [
            UserRoles.PLATFORM_ADMIN,
            UserRoles.SUPER_ADMIN,
        ]);
    });

    it('logs admin out with the admin role', async () => {
        await service.logout(request, response);

        expect(authService.logout).toHaveBeenCalledWith(request, response, UserRoles.PLATFORM_ADMIN);
    });

    it('refreshes admin token using the admin cookie', async () => {
        await service.refreshAdminToken(response, request);

        expect(authService.refreshAccessToken).toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
        );
    });

    it('admin refresh reads only adminRefreshToken and ignores customer tokens', async () => {
        request.cookies = {
            [COOKIE_NAMES.ADMIN_REFRESH_TOKEN]: 'admin-refresh',
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'customer-refresh',
        };

        await service.refreshAdminToken(response, request);

        expect(authService.refreshAccessToken).toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
        );
        expect(authService.refreshAccessToken).not.toHaveBeenCalledWith(
            response,
            request,
            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
        );
    });

    it('wraps refresh errors in InternalServerErrorException', async () => {
        authService.refreshAccessToken.mockRejectedValue(new Error('fail'));

        await expect(service.refreshAdminToken(response, request)).rejects.toBeInstanceOf(
            InternalServerErrorException,
        );
    });

    it('rethrows UnauthorizedException from AuthService', async () => {
        const unauthorized = new UnauthorizedException();
        authService.refreshAccessToken.mockRejectedValue(unauthorized);

        await expect(service.refreshAdminToken(response, request)).rejects.toBe(unauthorized);
    });

    it('returns current admin identity via AuthService', () => {
        const result = service.getMe(request);

        expect(result).toEqual({ id: 'admin' });
        expect(authService.getMe).toHaveBeenCalledWith(request, UserRoles.PLATFORM_ADMIN);
    });
});
