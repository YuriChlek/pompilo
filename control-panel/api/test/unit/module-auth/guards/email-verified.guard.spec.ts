/* eslint-disable @typescript-eslint/unbound-method */
import { ExecutionContext, ForbiddenException } from '@nestjs/common';
import { EmailVerifiedGuard } from '@/module-auth/guards/email-verified.guard';
import { UserRepository } from '@/module-user/repository/user.repository';
import { buildUserEntity } from '../../../fixtures/users.fixtures';

describe('EmailVerifiedGuard', () => {
    let guard: EmailVerifiedGuard;
    let mockUserRepository: jest.Mocked<UserRepository>;

    beforeEach(() => {
        mockUserRepository = {
            findById: jest.fn(),
        } as unknown as jest.Mocked<UserRepository>;

        guard = new EmailVerifiedGuard(mockUserRepository);
    });

    const createMockContext = (request = {}) =>
        ({
            switchToHttp: () => ({
                getRequest: () => request,
            }),
        }) as unknown as ExecutionContext;

    it('should throw ForbiddenException if user is not authenticated', async () => {
        const context = createMockContext({ user: null });
        await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });

    it('should allow admin roles to pass immediately without checking db', async () => {
        const context = createMockContext({ user: { userId: 'admin-id', role: 'admin' } });
        const result = await guard.canActivate(context);
        expect(result).toBe(true);
        expect(mockUserRepository.findById).not.toHaveBeenCalled();
    });

    it('should throw ForbiddenException if user does not exist in database', async () => {
        const context = createMockContext({ user: { userId: 'user-id', role: 'user' } });
        mockUserRepository.findById.mockResolvedValue(null);

        await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
        expect(mockUserRepository.findById).toHaveBeenCalledWith('user-id');
    });

    it('should throw ForbiddenException if user email is not verified', async () => {
        const context = createMockContext({ user: { userId: 'user-id', role: 'user' } });
        const user = buildUserEntity({ id: 'user-id', emailVerifiedAt: null });
        mockUserRepository.findById.mockResolvedValue(user);

        await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
        expect(mockUserRepository.findById).toHaveBeenCalledWith('user-id');
    });

    it('should return true if user email is verified', async () => {
        const context = createMockContext({ user: { userId: 'user-id', role: 'user' } });
        const user = buildUserEntity({ id: 'user-id', emailVerifiedAt: new Date() });
        mockUserRepository.findById.mockResolvedValue(user);

        const result = await guard.canActivate(context);
        expect(result).toBe(true);
        expect(mockUserRepository.findById).toHaveBeenCalledWith('user-id');
    });
});
