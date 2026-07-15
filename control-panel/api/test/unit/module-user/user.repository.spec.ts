/* eslint-disable @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return, @typescript-eslint/require-await */
import { UserRepository } from '@/module-user/repository/user.repository';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import * as schema from '@/module-drizzle/schemas';
import { NewUserModel } from '@/module-user/types/user.types';
import { UserRoles } from '@/module-auth/enums/auth.enums';

describe('UserRepository', () => {
    let repository: UserRepository;
    let mockDb: any;

    beforeEach(() => {
        mockDb = {
            insert: jest.fn().mockReturnThis(),
            values: jest.fn().mockReturnThis(),
            returning: jest.fn(),
            select: jest.fn().mockReturnThis(),
            from: jest.fn().mockReturnThis(),
            where: jest.fn().mockReturnThis(),
            limit: jest.fn().mockReturnThis(),
            update: jest.fn().mockReturnThis(),
            set: jest.fn().mockReturnThis(),
            transaction: jest.fn(),
        };

        repository = new UserRepository(mockDb as unknown as NodePgDatabase<typeof schema>);
    });

    describe('save', () => {
        it('normalizes email to lowercase before inserting', async () => {
            const user: NewUserModel = {
                name: 'john',
                email: 'JOHN@EXAMPLE.com',
                password: 'password',
                role: UserRoles.USER,
            };

            mockDb.returning.mockResolvedValue([{ id: '1', ...user, email: 'john@example.com' }]);

            const result = await repository.save(user);

            expect(mockDb.insert).toHaveBeenCalled();
            expect(mockDb.values).toHaveBeenCalledWith(
                expect.objectContaining({
                    email: 'john@example.com',
                }),
            );
            expect(result.email).toBe('john@example.com');
        });
    });

    describe('createWithTenant', () => {
        it('performs transactional creation of user, tenant, and membership', async () => {
            const user: NewUserModel = {
                name: 'john',
                email: 'JOHN@EXAMPLE.com',
                password: 'password',
                role: UserRoles.USER,
            };

            const mockTx = {
                insert: jest.fn().mockReturnThis(),
                values: jest.fn().mockReturnThis(),
                returning: jest.fn(),
            };

            mockDb.transaction.mockImplementation(async (callback: any) => {
                return callback(mockTx);
            });

            mockTx.returning
                .mockResolvedValueOnce([{ id: 'user-id', ...user, email: 'john@example.com' }])
                .mockResolvedValueOnce([{ id: 'tenant-id', name: 'My Tenant' }])
                .mockResolvedValueOnce([
                    { userId: 'user-id', tenantId: 'tenant-id', role: 'OWNER' },
                ]);

            const result = await repository.createWithTenant(user, 'My Tenant');

            expect(mockDb.transaction).toHaveBeenCalled();
            expect(mockTx.insert).toHaveBeenCalledTimes(3);
            expect(result.user.id).toBe('user-id');
            expect(result.tenant.id).toBe('tenant-id');
            expect(result.membership.role).toBe('OWNER');
        });
    });
});
