export type ParsedArgv = {
    commandName: string;
    args: string[];
    options: Record<string, string | boolean>;
};

export function parseCliArgv(argv: string[]): ParsedArgv {
    if (argv.length === 0) {
        return {
            commandName: '',
            args: [],
            options: {},
        };
    }

    const commandName = argv[0].trim();
    const rawArgsAndFlags = argv.slice(1);
    const args: string[] = [];
    const options: Record<string, string | boolean> = {};

    for (const item of rawArgsAndFlags) {
        const trimmed = item.trim();
        if (trimmed.startsWith('--')) {
            const clean = trimmed.slice(2);
            const eqIndex = clean.indexOf('=');
            if (eqIndex !== -1) {
                const key = clean.slice(0, eqIndex).trim();
                const value = clean.slice(eqIndex + 1);
                options[key] = value;
            } else {
                options[clean.trim()] = true;
            }
        } else if (trimmed) {
            args.push(trimmed);
        }
    }

    return {
        commandName,
        args,
        options,
    };
}
