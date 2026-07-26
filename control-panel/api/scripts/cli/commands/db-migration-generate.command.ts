import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { generateMigration } from '../operations/generate-migration';
import { rejectBooleanOption, rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

const allowedOptions = new Set(['name']);

export class DbMigrationGenerateCommand implements CliCommand {
    public readonly name = 'db:migration:generate';
    public readonly description =
        'Generates a new database migration and appends custom raw PG SQL';
    public readonly usage = 'db:migration:generate [--name=<migration-name>]';
    public readonly examples = [
        'npm run cli -- db:migration:generate',
        'npm run cli -- db:migration:generate --name=add-user-avatar',
    ];

    public run(context: CliCommandContext): CliCommandResult {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            for (const key of Object.keys(context.options)) {
                if (!allowedOptions.has(key)) {
                    context.logger.error(`Unsupported option for db:migration:generate: --${key}`);
                    context.logger.info(`Usage: ${this.usage}`);
                    return { exitCode: 1 };
                }
            }

            const invalidNameResult = rejectBooleanOption(context, 'name', this.usage);
            if (invalidNameResult) {
                return invalidNameResult;
            }

            const name = context.options.name;
            if (name === undefined) {
                generateMigration([]);
                return { exitCode: 0 };
            }

            if (typeof name !== 'string') {
                context.logger.error('Option --name must use --name=value format.');
                context.logger.info(`Usage: ${this.usage}`);
                return { exitCode: 1 };
            }

            generateMigration(['--name', name]);
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Migration generation failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
