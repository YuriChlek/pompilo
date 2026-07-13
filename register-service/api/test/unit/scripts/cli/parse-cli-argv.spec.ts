import { parseCliArgv } from '../../../../scripts/cli/parse-cli-argv';

describe('parseCliArgv', () => {
    it('should parse an empty argv array', () => {
        const result = parseCliArgv([]);
        expect(result.commandName).toBe('');
        expect(result.args).toEqual([]);
        expect(result.options).toEqual({});
    });

    it('should parse only commandName when no args/options are passed', () => {
        const result = parseCliArgv(['db:migration:run']);
        expect(result.commandName).toBe('db:migration:run');
        expect(result.args).toEqual([]);
        expect(result.options).toEqual({});
    });

    it('should parse commandName and positional args', () => {
        const result = parseCliArgv(['help', 'db:migration:run', 'extra-arg']);
        expect(result.commandName).toBe('help');
        expect(result.args).toEqual(['db:migration:run', 'extra-arg']);
        expect(result.options).toEqual({});
    });

    it('should parse key=value options', () => {
        const result = parseCliArgv([
            'admin:create',
            '--admin-email=test@test.com',
            '--role=superAdmin',
        ]);
        expect(result.commandName).toBe('admin:create');
        expect(result.args).toEqual([]);
        expect(result.options).toEqual({
            'admin-email': 'test@test.com',
            role: 'superAdmin',
        });
    });

    it('should parse boolean flags', () => {
        const result = parseCliArgv(['data-patches:push', '--force', '--dry-run']);
        expect(result.commandName).toBe('data-patches:push');
        expect(result.args).toEqual([]);
        expect(result.options).toEqual({
            force: true,
            'dry-run': true,
        });
    });

    it('should parse mixed positional args, key-value options, and flags', () => {
        const result = parseCliArgv([
            'db:migration:generate',
            'positional-val',
            '--name=add-user-table',
            '--verbose',
        ]);
        expect(result.commandName).toBe('db:migration:generate');
        expect(result.args).toEqual(['positional-val']);
        expect(result.options).toEqual({
            name: 'add-user-table',
            verbose: true,
        });
    });

    it('should trim arguments and ignore empty values', () => {
        const result = parseCliArgv(['   help   ', '   ', 'arg1']);
        expect(result.commandName).toBe('help');
        expect(result.args).toEqual(['arg1']);
    });
});
