import { CommandRegistry } from '../../../../scripts/cli/command-registry';
import { CliCommand } from '../../../../scripts/cli/cli-command.types';
import { createBuiltInCommandRegistry } from '../../../../scripts/cli';

describe('CommandRegistry', () => {
    let registry: CommandRegistry;

    beforeEach(() => {
        registry = new CommandRegistry();
    });

    const createMockCommand = (name: string): CliCommand => ({
        name,
        description: `Description for ${name}`,
        usage: `Usage for ${name}`,
        run: jest.fn().mockResolvedValue({ exitCode: 0 }),
    });

    it('should register and retrieve a command by name', () => {
        const cmd = createMockCommand('test:command');
        registry.register(cmd);

        expect(registry.get('test:command')).toBe(cmd);
    });

    it('should retrieve a command case-insensitively and with trimmed name', () => {
        const cmd = createMockCommand('test:command');
        registry.register(cmd);

        expect(registry.get('  TEST:command  ')).toBe(cmd);
    });

    it('should throw an error when registering a duplicate command name', () => {
        const cmd1 = createMockCommand('test:command');
        const cmd2 = createMockCommand('TEST:COMMAND');

        registry.register(cmd1);
        expect(() => registry.register(cmd2)).toThrow('Duplicate command registered: TEST:COMMAND');
    });

    it('should return undefined if a command is not registered', () => {
        expect(registry.get('non-existent')).toBeUndefined();
    });

    it('should retrieve all registered commands', () => {
        const cmd1 = createMockCommand('cmd1');
        const cmd2 = createMockCommand('cmd2');

        registry.register(cmd1);
        registry.register(cmd2);

        const all = registry.getAll();
        expect(all).toHaveLength(2);
        expect(all).toContain(cmd1);
        expect(all).toContain(cmd2);
    });

    it('should clear all registered commands', () => {
        const cmd = createMockCommand('test:command');
        registry.register(cmd);
        registry.clear();

        expect(registry.get('test:command')).toBeUndefined();
        expect(registry.getAll()).toHaveLength(0);
    });

    it('should expose operational commands through the built-in CLI registry', () => {
        const builtInRegistry = createBuiltInCommandRegistry();
        const expectedCommands = [
            'help',
            'data-patches:push',
            'data-patches:list',
            'data-patches:dry-run',
            'data-patches:create',
            'data-patches:build',
            'db:migration:generate',
            'db:migration:run',
            'db:dev:migration:run',
            'db:studio',
            'db:destructive-reset',
            'admin:create',
            'demo-data:push',
            'code:check-cycles',
        ];

        expect(
            builtInRegistry
                .getAll()
                .map(command => command.name)
                .sort(),
        ).toEqual([...expectedCommands].sort());

        for (const commandName of expectedCommands) {
            expect(builtInRegistry.get(commandName)?.name).toBe(commandName);
        }
    });
});
