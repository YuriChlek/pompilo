import { runCli } from '../../../../scripts/cli';
import { runDataPatches } from '../../../../scripts/cli/operations/run-data-patches';
import { createDataPatchFile } from '../../../../scripts/cli/operations/create-data-patch';
import { buildDataPatches } from '../../../../scripts/cli/operations/build-data-patches';

jest.mock('../../../../scripts/cli/operations/run-data-patches', () => ({
    runDataPatches: jest.fn().mockResolvedValue({ total: 0, skipped: 0, applied: 0 }),
}));

jest.mock('../../../../scripts/cli/operations/create-data-patch', () => ({
    createDataPatchFile: jest.fn().mockReturnValue('/path/to/patch.ts'),
}));

jest.mock('../../../../scripts/cli/operations/build-data-patches', () => ({
    buildDataPatches: jest.fn(),
}));

describe('Data Patches CLI Commands', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('data-patches:push should call runDataPatches with empty array', async () => {
        const exitCode = await runCli(['data-patches:push']);
        expect(exitCode).toBe(0);
        expect(runDataPatches).toHaveBeenCalledWith([]);
    });

    it('data-patches:list should call runDataPatches with ["--list"]', async () => {
        const exitCode = await runCli(['data-patches:list']);
        expect(exitCode).toBe(0);
        expect(runDataPatches).toHaveBeenCalledWith(['--list']);
    });

    it('data-patches:dry-run should call runDataPatches with ["--dry-run"]', async () => {
        const exitCode = await runCli(['data-patches:dry-run']);
        expect(exitCode).toBe(0);
        expect(runDataPatches).toHaveBeenCalledWith(['--dry-run']);
    });

    it('data-patches:create should call createDataPatchFile with parsed --name option', async () => {
        const exitCode = await runCli(['data-patches:create', '--name=my-cool-patch']);
        expect(exitCode).toBe(0);
        expect(createDataPatchFile).toHaveBeenCalledWith(['--name=my-cool-patch']);
    });

    it('data-patches:create should reject positional patch names', async () => {
        const exitCode = await runCli(['data-patches:create', 'my-cool-patch']);
        expect(exitCode).toBe(1);
        expect(createDataPatchFile).not.toHaveBeenCalled();
    });

    it('data-patches:create should reject space-separated --name values', async () => {
        const exitCode = await runCli(['data-patches:create', '--name', 'my-cool-patch']);
        expect(exitCode).toBe(1);
        expect(createDataPatchFile).not.toHaveBeenCalled();
    });

    it('data-patches:create should call createDataPatchFile with empty array if no name is provided', async () => {
        const exitCode = await runCli(['data-patches:create']);
        expect(exitCode).toBe(0);
        expect(createDataPatchFile).toHaveBeenCalledWith([]);
    });

    it('data-patches:build should call buildDataPatches', async () => {
        const exitCode = await runCli(['data-patches:build']);
        expect(exitCode).toBe(0);
        expect(buildDataPatches).toHaveBeenCalled();
    });

    it('data-patches:push should reject positional arguments', async () => {
        const exitCode = await runCli(['data-patches:push', 'unexpected']);
        expect(exitCode).toBe(1);
        expect(runDataPatches).not.toHaveBeenCalled();
    });
});
