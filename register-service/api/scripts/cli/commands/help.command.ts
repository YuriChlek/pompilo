import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { CommandRegistry } from '../command-registry';

export class HelpCommand implements CliCommand {
    public readonly name = 'help';
    public readonly description = 'Shows help information for commands';
    public readonly usage = 'help [command]';
    public readonly examples = ['npm run cli -- help', 'npm run cli -- help db:migration:run'];

    constructor(private readonly registry: CommandRegistry) {}

    public run(context: CliCommandContext): CliCommandResult {
        const { args, logger } = context;

        if (args.length > 0) {
            const targetCommandName = args[0];
            const targetCommand = this.registry.get(targetCommandName);

            if (!targetCommand) {
                logger.error(`Unknown command to show help for: ${targetCommandName}`);
                return { exitCode: 1 };
            }

            logger.info(`Command:     ${targetCommand.name}`);
            logger.info(`Description: ${targetCommand.description}`);
            logger.info(`Usage:       ${targetCommand.usage}`);
            if (targetCommand.examples && targetCommand.examples.length > 0) {
                logger.info('Examples:');
                for (const example of targetCommand.examples) {
                    logger.info(`  ${example}`);
                }
            }
            return { exitCode: 0 };
        }

        logger.info('Pampilo operational CLI');
        logger.info('Usage: npm run cli -- <command> [options]');
        logger.info('');
        logger.info('Available commands:');

        const commands = this.registry.getAll();
        if (commands.length === 0) {
            logger.info('  (No commands registered)');
            return { exitCode: 0 };
        }

        const maxLength = Math.max(...commands.map(c => c.name.length), 0);
        for (const cmd of commands) {
            const paddedName = cmd.name.padEnd(maxLength + 2, ' ');
            logger.info(`  ${paddedName}${cmd.description}`);
        }

        return { exitCode: 0 };
    }
}
