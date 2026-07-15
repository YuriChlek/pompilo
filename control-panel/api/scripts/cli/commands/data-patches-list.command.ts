import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { runDataPatches } from '../operations/run-data-patches';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DataPatchesListCommand implements CliCommand {
    public readonly name = 'data-patches:list';
    public readonly description = 'Lists all discovered data patches and their status';
    public readonly usage = 'data-patches:list';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            await runDataPatches(['--list']);
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Failed to list data patches: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
