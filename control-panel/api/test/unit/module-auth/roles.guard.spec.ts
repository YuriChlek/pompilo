import { Reflector } from '@nestjs/core';
import { ExecutionContext, ForbiddenException } from '@nestjs/common';
import { RolesGuard } from '@/module-auth/guards/roles.guard';
import { UserRoles } from '@/module-auth/enums/auth.enums';

describe('RolesGuard', () => {
    let guard: RolesGuard;
    let reflector: {
        getAllAndOverride: jest.Mock;
    };

    beforeEach(() => {
        reflector = {
            getAllAndOverride: jest.fn(),
        };
        guard = new RolesGuard(reflector as unknown as Reflector);
    });

    const createMockContext = (request = {}) =>
        ({
            switchToHttp: () => ({
                getRequest: () => request,
            }),
            getHandler: () => ({}),
            getClass: () => ({}),
        }) as unknown as ExecutionContext;

    it('returns true if no roles are required', () => {
        reflector.getAllAndOverride.mockReturnValue(undefined);
        const context = createMockContext();

        const result = guard.canActivate(context);
        expect(result).toBe(true);
    });

    it('returns true if required roles empty', () => {
        reflector.getAllAndOverride.mockReturnValue([]);
        const context = createMockContext();

        const result = guard.canActivate(context);
        expect(result).toBe(true);
    });

    it('throws ForbiddenException if user is not on request', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.USER]);
        const context = createMockContext({});

        expect(() => guard.canActivate(context)).toThrow(ForbiddenException);
    });

    it('throws ForbiddenException if user has no role', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.USER]);
        const context = createMockContext({ user: { realm: 'customer' } });

        expect(() => guard.canActivate(context)).toThrow(ForbiddenException);
    });

    it('throws ForbiddenException on role/realm mismatch', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.USER]);

        // Athlete role with admin realm -> mismatch!
        const context = createMockContext({
            user: {
                role: UserRoles.USER,
                realm: 'admin',
            },
        });

        expect(() => guard.canActivate(context)).toThrow('Realm and role mismatch.');
    });

    it('throws ForbiddenException on role/realm mismatch for admin', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.PLATFORM_ADMIN]);

        // Admin role with customer realm -> mismatch!
        const context = createMockContext({
            user: {
                role: UserRoles.PLATFORM_ADMIN,
                realm: 'customer',
            },
        });

        expect(() => guard.canActivate(context)).toThrow('Realm and role mismatch.');
    });

    it('returns true if user realm matches role realm and role is allowed', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.USER]);

        const context = createMockContext({
            user: {
                role: UserRoles.USER,
                realm: 'customer',
            },
        });

        const result = guard.canActivate(context);
        expect(result).toBe(true);
    });

    it('throws ForbiddenException if user has correct realm but role is not allowed', () => {
        reflector.getAllAndOverride.mockReturnValue([UserRoles.PLATFORM_ADMIN]);

        const context = createMockContext({
            user: {
                role: UserRoles.USER,
                realm: 'customer',
            },
        });

        expect(() => guard.canActivate(context)).toThrow('Access denied.');
    });
});
