import { CliCommandContext, CliCommandResult } from './cli-command.types';

export function rejectUnexpectedArgs(context: CliCommandContext): CliCommandResult | null {
    if (context.args.length === 0) {
        return null;
    }

    context.logger.error(
        `Unexpected positional arguments for ${context.commandName}: ${context.args.join(', ')}`,
    );
    return { exitCode: 1 };
}

export function rejectBooleanOption(
    context: CliCommandContext,
    optionName: string,
    usage: string,
): CliCommandResult | null {
    if (context.options[optionName] !== true) {
        return null;
    }

    context.logger.error(`Option --${optionName} must use --${optionName}=value format.`);
    context.logger.info(`Usage: ${usage}`);
    return { exitCode: 1 };
}
