import { runCli } from '../../../../scripts/cli';
import { generateMigration } from '../../../../scripts/cli/operations/generate-migration';
import { runDestructiveReset } from '../../../../scripts/cli/operations/destructive-reset';
import { spawnSync, spawn } from 'node:child_process';

jest.mock('../../../../scripts/cli/operations/generate-migration', () => ({
    generateMigration: jest.fn(),
}));

jest.mock('../../../../scripts/cli/operations/destructive-reset', () => ({
    runDestructiveReset: jest.fn(),
}));

jest.mock('node:child_process', () => ({
    spawnSync: jest.fn().mockReturnValue({ status: 0 }),
    spawn: jest.fn().mockReturnValue({
        pid: 123,
        killed: false,
        kill: jest.fn(),
        on: jest.fn((event: string, cb: (code: number) => void) => {
            if (event === 'close') {
                cb(0);
            }
        }),
        off: jest.fn(),
    }),
}));

describe('DB CLI Commands', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('db:migration:generate should call generateMigration without args when --name is missing', async () => {
        const exitCode = await runCli(['db:migration:generate']);
        expect(exitCode).toBe(0);
        expect(generateMigration).toHaveBeenCalledWith([]);
    });

    it('db:migration:generate should call generateMigration with name', async () => {
        const exitCode = await runCli(['db:migration:generate', '--name=add-users']);
        expect(exitCode).toBe(0);
        expect(generateMigration).toHaveBeenCalledWith(['--name', 'add-users']);
    });

    it('db:migration:generate should reject positional migration names', async () => {
        const exitCode = await runCli(['db:migration:generate', 'add-users']);
        expect(exitCode).toBe(1);
        expect(generateMigration).not.toHaveBeenCalled();
    });

    it('db:migration:generate should reject space-separated --name values', async () => {
        const exitCode = await runCli(['db:migration:generate', '--name', 'add-users']);
        expect(exitCode).toBe(1);
        expect(generateMigration).not.toHaveBeenCalled();
    });

    it('db:migration:generate should reject unsupported options', async () => {
        const exitCode = await runCli(['db:migration:generate', '--foo=bar']);
        expect(exitCode).toBe(1);
        expect(generateMigration).not.toHaveBeenCalled();
    });

    it('db:migration:run should run drizzle-kit migrate', async () => {
        const exitCode = await runCli(['db:migration:run']);
        expect(exitCode).toBe(0);

        expect(spawnSync).toHaveBeenCalled();
        const mockCalls = (spawnSync as jest.Mock).mock.calls as [
            string,
            string[],
            Record<string, unknown>,
        ][];
        const lastCall = mockCalls[mockCalls.length - 1];
        expect(lastCall[0]).toBe('npx');
        expect(lastCall[1]).toEqual(['drizzle-kit', 'migrate']);
    });

    it('db:migration:run should reject positional arguments', async () => {
        const exitCode = await runCli(['db:migration:run', 'unexpected']);
        expect(exitCode).toBe(1);
        expect(spawnSync).not.toHaveBeenCalled();
    });

    it('db:dev:migration:run should run drizzle-kit migrate with NODE_ENV=development', async () => {
        const exitCode = await runCli(['db:dev:migration:run']);
        expect(exitCode).toBe(0);

        expect(spawnSync).toHaveBeenCalled();
        const mockCalls = (spawnSync as jest.Mock).mock.calls as [
            string,
            string[],
            { env?: Record<string, string> },
        ][];
        const lastCall = mockCalls[mockCalls.length - 1];
        expect(lastCall[0]).toBe('npx');
        expect(lastCall[1]).toEqual(['drizzle-kit', 'migrate']);

        // Assert NODE_ENV is development
        const options = lastCall[2];
        expect(options?.env?.NODE_ENV).toBe('development');
    });

    it('db:destructive-reset should call runDestructiveReset', async () => {
        const exitCode = await runCli(['db:destructive-reset']);
        expect(exitCode).toBe(0);
        expect(runDestructiveReset).toHaveBeenCalled();
    });

    it('db:studio should spawn drizzle-kit studio', async () => {
        const exitCode = await runCli(['db:studio']);
        expect(exitCode).toBe(0);

        expect(spawn).toHaveBeenCalled();
        const mockCalls = (spawn as jest.Mock).mock.calls as [
            string,
            string[],
            Record<string, unknown>,
        ][];
        const lastCall = mockCalls[mockCalls.length - 1];
        expect(lastCall[0]).toBe('npx');
        expect(lastCall[1]).toEqual(['drizzle-kit', 'studio']);
    });
});
