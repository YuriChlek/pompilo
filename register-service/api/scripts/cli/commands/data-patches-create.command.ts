import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { createDataPatchFile } from '../operations/create-data-patch';
import { rejectBooleanOption, rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DataPatchesCreateCommand implements CliCommand {
    public readonly name = 'data-patches:create';
    public readonly description = 'Creates a new template file for data patches';
    public readonly usage = 'data-patches:create --name=<patch-name>';
    public readonly examples = ['npm run cli -- data-patches:create --name=identity-bootstrap'];

    public run(context: CliCommandContext): CliCommandResult {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            const invalidNameResult = rejectBooleanOption(context, 'name', this.usage);
            if (invalidNameResult) {
                return invalidNameResult;
            }

            const name = context.options.name;
            const argv: string[] = [];
            if (typeof name === 'string') {
                argv.push(`--name=${name}`);
            }
            const filePath = createDataPatchFile(argv);
            context.logger.info(`Created data patch: ${filePath}`);
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Failed to create data patch: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
