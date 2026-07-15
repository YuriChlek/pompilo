import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { ConfigService } from '@nestjs/config';
import { Request, Response } from 'express';
import { COOKIE_NAMES } from '@/module-auth/enums/auth.enums';

describe('DeviceIdService', () => {
    let service: DeviceIdService;
    let configService: {
        get: jest.Mock;
        getOrThrow: jest.Mock;
    };
    let mockResponse: {
        cookie: jest.Mock;
    };

    beforeEach(() => {
        configService = {
            get: jest.fn().mockImplementation((key: string) => {
                if (key === 'NODE_ENV') return 'development';
                if (key === 'COOKIE_DOMAIN') return 'localhost';
                return undefined;
            }),
            getOrThrow: jest.fn().mockImplementation((key: string) => {
                if (key === 'DEVICE_ID_COOKIE_TTL') return '365d';
                return undefined;
            }),
        };

        mockResponse = {
            cookie: jest.fn(),
        };

        service = new DeviceIdService(configService as unknown as ConfigService);
    });

    describe('validateUuid', () => {
        it('should return true for valid UUIDv4', () => {
            expect(service.validateUuid('a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d')).toBe(true);
            expect(service.validateUuid('00000000-0000-4000-8000-000000000000')).toBe(true);
        });

        it('should return false for invalid UUIDs', () => {
            expect(service.validateUuid('')).toBe(false);
            expect(service.validateUuid('not-a-uuid')).toBe(false);
            expect(service.validateUuid('a1b2c3d4-e5f6-6a8b-9c0d-1e2f3a4b5c6d')).toBe(false); // wrong version (6)
            expect(service.validateUuid('a1b2c3d4-e5f6-4a8b-5c0d-1e2f3a4b5c6d')).toBe(false); // wrong variant (5)
        });
    });

    describe('readDeviceId', () => {
        it('should read from raw cookies if present and valid', () => {
            const uuid = 'a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d';
            const mockRequest = {
                cookies: { [COOKIE_NAMES.DEVICE_ID]: uuid },
                headers: {},
            } as unknown as Request;

            const result = service.readDeviceId(mockRequest);

            expect(result).toBe(uuid);
        });

        it('should read and parse from cookie header if present and valid', () => {
            const uuid = 'a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d';
            const mockRequest = {
                cookies: {},
                headers: {
                    cookie: `${COOKIE_NAMES.DEVICE_ID}=${uuid}; otherCookie=value`,
                },
            } as unknown as Request;

            const result = service.readDeviceId(mockRequest);

            expect(result).toBe(uuid);
        });

        it('should return null if cookie is missing', () => {
            const mockRequest = {
                cookies: {},
                headers: {},
            } as unknown as Request;

            const result = service.readDeviceId(mockRequest);

            expect(result).toBeNull();
        });

        it('should return null if cookie is invalid UUID', () => {
            const mockRequest = {
                cookies: { [COOKIE_NAMES.DEVICE_ID]: 'invalid-uuid' },
                headers: {},
            } as unknown as Request;

            const result = service.readDeviceId(mockRequest);

            expect(result).toBeNull();
        });
    });

    describe('getOrCreateDeviceId', () => {
        it('should return existing device ID and isNew: false if present and valid', () => {
            const uuid = 'a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d';
            const mockRequest = {
                cookies: { [COOKIE_NAMES.DEVICE_ID]: uuid },
                headers: {},
            } as unknown as Request;

            const result = service.getOrCreateDeviceId(mockRequest);

            expect(result).toEqual({ deviceId: uuid, isNew: false });
        });

        it('should generate new device ID and isNew: true if cookie is missing', () => {
            const mockRequest = {
                cookies: {},
                headers: {},
            } as unknown as Request;

            const result = service.getOrCreateDeviceId(mockRequest);

            expect(result.isNew).toBe(true);
            expect(service.validateUuid(result.deviceId)).toBe(true);
        });

        it('should generate new device ID and isNew: true if cookie is invalid', () => {
            const mockRequest = {
                cookies: { [COOKIE_NAMES.DEVICE_ID]: 'invalid-uuid' },
                headers: {},
            } as unknown as Request;

            const result = service.getOrCreateDeviceId(mockRequest);

            expect(result.isNew).toBe(true);
            expect(service.validateUuid(result.deviceId)).toBe(true);
        });
    });

    describe('setDeviceIdCookie', () => {
        it('should call response.cookie with correct options in development', () => {
            const uuid = 'a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d';

            service.setDeviceIdCookie(mockResponse as unknown as Response, uuid);

            expect(mockResponse.cookie).toHaveBeenCalledWith(COOKIE_NAMES.DEVICE_ID, uuid, {
                httpOnly: true,
                secure: false,
                sameSite: 'lax',
                path: '/',
                maxAge: 365 * 24 * 60 * 60 * 1000, // 365d in ms
                domain: 'localhost',
            });
        });

        it('should call response.cookie with secure: true in production', () => {
            configService.get.mockImplementation((key: string) => {
                if (key === 'NODE_ENV') return 'production';
                return undefined;
            });
            const uuid = 'a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d';

            service.setDeviceIdCookie(mockResponse as unknown as Response, uuid);

            expect(mockResponse.cookie).toHaveBeenCalledWith(COOKIE_NAMES.DEVICE_ID, uuid, {
                httpOnly: true,
                secure: true,
                sameSite: 'lax',
                path: '/',
                maxAge: 365 * 24 * 60 * 60 * 1000,
            });
        });
    });
});
