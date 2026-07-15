import { runCli } from '../../../../scripts/cli';
import { checkCircularDependencies } from '../../../../scripts/cli/operations/check-circular-dependencies';

jest.mock('../../../../scripts/cli/operations/check-circular-dependencies', () => ({
    checkCircularDependencies: jest.fn(),
    resolveCliProjectRoot: jest.fn(() => '/repo/api'),
}));

describe('Code CLI Commands', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        jest.mocked(checkCircularDependencies).mockReturnValue(0);
    });

    it('code:check-cycles should run madge against src by default', async () => {
        const exitCode = await runCli(['code:check-cycles']);

        expect(exitCode).toBe(0);
        expect(checkCircularDependencies).toHaveBeenCalledWith(
            expect.objectContaining({
                projectRoot: '/repo/api',
                targetPath: 'src',
            }),
        );
    });

    it('code:check-cycles should run madge against a custom path', async () => {
        const exitCode = await runCli(['code:check-cycles', '--path=src/module-auth']);

        expect(exitCode).toBe(0);
        expect(checkCircularDependencies).toHaveBeenCalledWith(
            expect.objectContaining({
                targetPath: 'src/module-auth',
            }),
        );
    });

    it('code:check-cycles should reject positional arguments', async () => {
        const exitCode = await runCli(['code:check-cycles', 'src']);

        expect(exitCode).toBe(1);
        expect(checkCircularDependencies).not.toHaveBeenCalled();
    });

    it('code:check-cycles should reject space-separated path values', async () => {
        const exitCode = await runCli(['code:check-cycles', '--path', 'src']);

        expect(exitCode).toBe(1);
        expect(checkCircularDependencies).not.toHaveBeenCalled();
    });

    it('code:check-cycles should reject unsupported options', async () => {
        const exitCode = await runCli(['code:check-cycles', '--foo=bar']);

        expect(exitCode).toBe(1);
        expect(checkCircularDependencies).not.toHaveBeenCalled();
    });

    it('code:check-cycles should return madge exit code when cycles are found', async () => {
        jest.mocked(checkCircularDependencies).mockReturnValue(1);

        const exitCode = await runCli(['code:check-cycles']);

        expect(exitCode).toBe(1);
    });
});
