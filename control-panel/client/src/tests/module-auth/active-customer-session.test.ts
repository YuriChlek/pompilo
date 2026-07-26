import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getActiveCustomerSession } from '@/features/module-auth/server/active-customer-session';
import { resolveCustomerSessionState } from '@/features/module-auth/server/customer-session-resolver';
import { CustomerSessionState } from '@/features/module-auth/enums/customer-session-state.enums';
import { getCurrentUser } from '@/features/module-auth/api-service/server';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';

vi.mock('next/headers', () => ({
    cookies: vi.fn().mockResolvedValue({}),
}));

vi.mock('@/features/module-auth/server/customer-session-resolver', () => ({
    resolveCustomerSessionState: vi.fn(),
}));

vi.mock('@/features/module-auth/api-service/server', () => ({
    getCurrentUser: vi.fn(),
}));

describe('getActiveCustomerSession', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should return null session for NONE state', async () => {
        vi.mocked(resolveCustomerSessionState).mockReturnValue(CustomerSessionState.NONE);

        const session = await getActiveCustomerSession();

        expect(session).toEqual({ user: null, role: null });
        expect(getCurrentUser).not.toHaveBeenCalled();
    });

    it('should return user session when state is CUSTOMER and API returns user', async () => {
        vi.mocked(resolveCustomerSessionState).mockReturnValue(CustomerSessionState.CUSTOMER);
        const mockUser = { id: '1', role: UserRoles.USER };
        vi.mocked(getCurrentUser).mockResolvedValue(mockUser as unknown as User);

        const session = await getActiveCustomerSession();

        expect(session).toEqual({ user: mockUser, role: UserRoles.USER });
        expect(getCurrentUser).toHaveBeenCalledWith();
    });

    it('should return another user session when state is CUSTOMER and API returns user', async () => {
        vi.mocked(resolveCustomerSessionState).mockReturnValue(CustomerSessionState.CUSTOMER);
        const mockUser = { id: '2', role: UserRoles.USER };
        vi.mocked(getCurrentUser).mockResolvedValue(mockUser as unknown as User);

        const session = await getActiveCustomerSession();

        expect(session).toEqual({ user: mockUser, role: UserRoles.USER });
        expect(getCurrentUser).toHaveBeenCalledWith();
    });

    it('should return null if user role is not a customer user', async () => {
        vi.mocked(resolveCustomerSessionState).mockReturnValue(CustomerSessionState.CUSTOMER);
        const mockUser = { id: '3', role: UserRoles.PLATFORM_ADMIN };
        vi.mocked(getCurrentUser).mockResolvedValue(mockUser as unknown as User);

        const session = await getActiveCustomerSession();

        expect(session).toEqual({ user: null, role: null });
    });

    it('should rethrow technical errors from getCurrentUser (Error Boundary trigger)', async () => {
        vi.mocked(resolveCustomerSessionState).mockReturnValue(CustomerSessionState.CUSTOMER);
        const techError = new Error('Technical Connection Error');
        vi.mocked(getCurrentUser).mockRejectedValue(techError);

        await expect(getActiveCustomerSession()).rejects.toThrow('Technical Connection Error');
    });
});
