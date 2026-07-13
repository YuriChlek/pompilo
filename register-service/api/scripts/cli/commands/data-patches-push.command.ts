import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { runDataPatches } from '../operations/run-data-patches';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DataPatchesPushCommand implements CliCommand {
    public readonly name = 'data-patches:push';
    public readonly description = 'Executes pending idempotent data patches in transaction';
    public readonly usage = 'data-patches:push';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            // We pass an empty array to runDataPatches to avoid listing or dry-running
            await runDataPatches([]);
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Failed to push data patches: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
