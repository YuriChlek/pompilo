import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { runDestructiveReset } from '../operations/destructive-reset';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DbDestructiveResetCommand implements CliCommand {
    public readonly name = 'db:destructive-reset';
    public readonly description = 'Drops public schema and recreates, runs migrations, seeds data';
    public readonly usage = 'db:destructive-reset';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            await runDestructiveReset();
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Destructive reset failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
