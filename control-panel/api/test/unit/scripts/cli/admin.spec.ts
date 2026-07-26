import { runCli } from '../../../../scripts/cli';
import { createAdminUser } from '../../../../scripts/cli/operations/create-admin-user';

jest.mock('../../../../scripts/cli/operations/create-admin-user', () => ({
    createAdminUser: jest.fn().mockResolvedValue({
        userId: 'admin-id-123',
        name: 'Admin Name',
        email: 'admin@example.com',
        role: 'platformAdmin',
    }),
}));

describe('Admin CLI Commands', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('admin:create should call createAdminUser with parsed named options', async () => {
        const exitCode = await runCli([
            'admin:create',
            '--admin-email=admin@example.com',
            '--admin-password=Secure123456',
            '--admin-firstname=John',
            '--admin-lastname=Doe',
            '--role=superAdmin',
        ]);

        expect(exitCode).toBe(0);
        expect(createAdminUser).toHaveBeenCalledWith({
            'admin-email': 'admin@example.com',
            'admin-password': 'Secure123456',
            'admin-firstname': 'John',
            'admin-lastname': 'Doe',
            role: 'superAdmin',
        });
    });

    it('admin:create should reject backward-compatible aliases', async () => {
        const exitCode = await runCli([
            'admin:create',
            '--email=admin@example.com',
            '--password=Secure123456',
            '--name=John Doe',
        ]);

        expect(exitCode).toBe(1);
        expect(createAdminUser).not.toHaveBeenCalled();
    });

    it('admin:create should reject positional values from space-separated flags', async () => {
        const exitCode = await runCli([
            'admin:create',
            '--admin-email',
            'admin@example.com',
            '--admin-password=Secure123456',
            '--admin-firstname=John',
            '--admin-lastname=Doe',
        ]);

        expect(exitCode).toBe(1);
        expect(createAdminUser).not.toHaveBeenCalled();
    });

    it('admin:create should require admin-prefixed name flags', async () => {
        const exitCode = await runCli([
            'admin:create',
            '--admin-email=admin@example.com',
            '--admin-password=Secure123456',
            '--name=John Doe',
        ]);

        expect(exitCode).toBe(1);
        expect(createAdminUser).not.toHaveBeenCalled();
    });

    it('admin:create should return 1 and print error if createAdminUser throws error', async () => {
        (createAdminUser as jest.Mock).mockRejectedValueOnce(new Error('User already exists'));

        const exitCode = await runCli([
            'admin:create',
            '--admin-email=admin@example.com',
            '--admin-password=Secure123456',
            '--admin-firstname=John',
            '--admin-lastname=Doe',
        ]);

        expect(exitCode).toBe(1);
    });
});
