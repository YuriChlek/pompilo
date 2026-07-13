import type { Request, Response } from 'express';
import { METHOD_METADATA, PATH_METADATA } from '@nestjs/common/constants';
import { RequestMethod } from '@nestjs/common';
import { AdminAuthController } from '@/module-admin-auth/controllers/admin-auth.controller';
import { AdminAuthService } from '@/module-admin-auth/services/admin-auth.service';
import { LoginAdminDto } from '@/module-admin-auth/dto/login-admin.dto';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';

describe('AdminAuthController', () => {
    let controller: AdminAuthController;
    let service: jest.Mocked<AdminAuthService>;
    let loginMock: jest.MockedFunction<AdminAuthService['login']>;
    let resendLoginCheckpointMock: jest.MockedFunction<AdminAuthService['resendLoginCheckpoint']>;
    let logoutMock: jest.MockedFunction<AdminAuthService['logout']>;
    let refreshMock: jest.MockedFunction<AdminAuthService['refreshAdminToken']>;
    let getMeMock: jest.MockedFunction<AdminAuthService['getMe']>;

    beforeEach(() => {
        loginMock = jest.fn();
        resendLoginCheckpointMock = jest.fn();
        logoutMock = jest.fn();
        refreshMock = jest.fn();
        getMeMock = jest.fn();
        service = {
            login: loginMock,
            resendLoginCheckpoint: resendLoginCheckpointMock,
            logout: logoutMock,
            refreshAdminToken: refreshMock,
            getMe: getMeMock,
        } as jest.Mocked<AdminAuthService>;
        controller = new AdminAuthController(service);
    });

    it('delegates login to the admin auth service', async () => {
        const req = {} as Request;
        const res = {} as Response;
        const dto: LoginAdminDto = {
            login: 'admin@example.com',
            password: 'Password1',
            role: UserRoles.PLATFORM_ADMIN,
        };

        await controller.login(res, req, dto);

        expect(loginMock).toHaveBeenCalledWith(res, req, dto);
    });

    it('exposes admin POST /admin/checkpoint/resend and delegates to the service', async () => {
        const req = {} as Request;
        const res = {} as Response;
        const dto: ResendCheckpointDto = { checkpointToken: 'checkpoint-token' };

        await controller.resendCheckpoint(res, req, dto);

        const handler = Object.getOwnPropertyDescriptor(
            AdminAuthController.prototype,
            'resendCheckpoint',
        )!.value as (...args: unknown[]) => unknown;

        expect(Reflect.getMetadata(METHOD_METADATA, handler)).toBe(RequestMethod.POST);
        expect(Reflect.getMetadata(PATH_METADATA, handler)).toBe('checkpoint/resend');
        expect(resendLoginCheckpointMock).toHaveBeenCalledWith(res, req, dto);
    });

    it('exposes logout/refresh/me endpoints', async () => {
        const req = {} as Request;
        const res = {} as Response;

        await controller.logout(res, req);
        await controller.refreshAdminToken(res, req);
        controller.getMe(req);

        expect(logoutMock).toHaveBeenCalledWith(req, res);
        expect(refreshMock).toHaveBeenCalledWith(res, req);
        expect(getMeMock).toHaveBeenCalledWith(req);
    });
});
