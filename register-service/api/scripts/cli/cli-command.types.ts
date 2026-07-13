export type CliLogger = {
    info(message: string): void;
    warn(message: string): void;
    error(message: string): void;
};

export type CliCommandContext = {
    commandName: string;
    args: string[];
    options: Record<string, string | boolean>;
    env: NodeJS.ProcessEnv;
    cwd: string;
    logger: CliLogger;
};

export type CliCommandResult = {
    exitCode: number;
    error?: Error;
};

export type CliCommand = {
    name: string;
    description: string;
    usage: string;
    examples?: string[];
    run(context: CliCommandContext): CliCommandResult | Promise<CliCommandResult>;
};
