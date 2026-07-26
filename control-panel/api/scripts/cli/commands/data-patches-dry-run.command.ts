import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { runDataPatches } from '../operations/run-data-patches';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DataPatchesDryRunCommand implements CliCommand {
    public readonly name = 'data-patches:dry-run';
    public readonly description = 'Runs data patches simulation (dry-run mode)';
    public readonly usage = 'data-patches:dry-run';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            await runDataPatches(['--dry-run']);
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Failed to dry-run data patches: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
