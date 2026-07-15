import { runCli } from '../../../../scripts/cli';
import { CommandRegistry } from '../../../../scripts/cli/command-registry';
import { HelpCommand } from '../../../../scripts/cli/commands/help.command';
import { CliCommand } from '../../../../scripts/cli/cli-command.types';

describe('runCli Dispatcher and Help Command', () => {
    let registry: CommandRegistry;
    let mockLog: jest.SpyInstance;
    let mockWarn: jest.SpyInstance;
    let mockError: jest.SpyInstance;

    beforeEach(() => {
        registry = new CommandRegistry();
        mockLog = jest.spyOn(console, 'log').mockImplementation(() => {});
        mockWarn = jest.spyOn(console, 'warn').mockImplementation(() => {});
        mockError = jest.spyOn(console, 'error').mockImplementation(() => {});
    });

    afterEach(() => {
        mockLog.mockRestore();
        mockWarn.mockRestore();
        mockError.mockRestore();
    });

    const createMockCommand = (name: string, exitCode = 0, runMock?: jest.Mock): CliCommand => ({
        name,
        description: `Description for ${name}`,
        usage: `Usage for ${name}`,
        examples: [`npm run cli -- ${name} --opt`],
        run: runMock || jest.fn().mockResolvedValue({ exitCode }),
    });

    it('should return 1 and print error if no command is specified', async () => {
        // Register help command so it can fallback to help display
        registry.register(new HelpCommand(registry));
        const exitCode = await runCli([], registry);

        expect(exitCode).toBe(1);
        expect(mockError).toHaveBeenCalledWith(expect.stringContaining('No command specified'));
        expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Available commands'));
    });

    it('should return 1 and print error if an unknown command is passed', async () => {
        const exitCode = await runCli(['unknown:cmd'], registry);

        expect(exitCode).toBe(1);
        expect(mockError).toHaveBeenCalledWith(
            expect.stringContaining('Unknown command: unknown:cmd'),
        );
    });

    it('should dispatch to the registered command and return its exit code', async () => {
        const runMock = jest.fn().mockResolvedValue({ exitCode: 42 });
        const cmd = createMockCommand('test:cmd', 42, runMock);
        registry.register(cmd);

        const exitCode = await runCli(['test:cmd', 'arg1', '--foo=bar'], registry);

        expect(runMock).toHaveBeenCalledWith(
            expect.objectContaining({
                commandName: 'test:cmd',
                args: ['arg1'],
                options: { foo: 'bar' },
            }),
        );
        expect(exitCode).toBe(42);
    });

    it('should catch unhandled errors from commands, log them, and return 1', async () => {
        const runMock = jest.fn().mockRejectedValue(new Error('Something went wrong'));
        const cmd = createMockCommand('fail:cmd', 0, runMock);
        registry.register(cmd);

        const exitCode = await runCli(['fail:cmd'], registry);

        expect(exitCode).toBe(1);
        expect(mockError).toHaveBeenCalledWith(expect.stringContaining('Something went wrong'));
    });

    describe('HelpCommand Behavior', () => {
        beforeEach(() => {
            registry.register(new HelpCommand(registry));
        });

        it('should show list of commands including help itself', async () => {
            const cmd = createMockCommand('other:cmd');
            registry.register(cmd);

            const exitCode = await runCli(['help'], registry);

            expect(exitCode).toBe(0);
            expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Available commands'));
            expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('help  '));
            expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('other:cmd  '));
        });

        it('should show detailed help for a specific command', async () => {
            const cmd = createMockCommand('other:cmd');
            registry.register(cmd);

            const exitCode = await runCli(['help', 'other:cmd'], registry);

            expect(exitCode).toBe(0);
            expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Command:     other:cmd'));
            expect(mockLog).toHaveBeenCalledWith(
                expect.stringContaining('Description: Description for other:cmd'),
            );
            expect(mockLog).toHaveBeenCalledWith(
                expect.stringContaining('Usage:       Usage for other:cmd'),
            );
            expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Examples:'));
            expect(mockLog).toHaveBeenCalledWith(
                expect.stringContaining('npm run cli -- other:cmd --opt'),
            );
        });

        it('should return 1 if help requested for an unknown command', async () => {
            const exitCode = await runCli(['help', 'non-existent'], registry);

            expect(exitCode).toBe(1);
            expect(mockError).toHaveBeenCalledWith(
                expect.stringContaining('Unknown command to show help for: non-existent'),
            );
        });
    });
});
