import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { buildDataPatches } from '../operations/build-data-patches';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DataPatchesBuildCommand implements CliCommand {
    public readonly name = 'data-patches:build';
    public readonly description = 'Compiles data patches and generates a manifest';
    public readonly usage = 'data-patches:build';

    public run(context: CliCommandContext): CliCommandResult {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            buildDataPatches();
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Failed to build data patches: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
