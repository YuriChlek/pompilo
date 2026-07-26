import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { rejectBooleanOption, rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';
import {
    checkCircularDependencies,
    resolveCliProjectRoot,
} from '../operations/check-circular-dependencies';

const allowedOptions = new Set(['path']);

export class CodeCheckCyclesCommand implements CliCommand {
    public readonly name = 'code:check-cycles';
    public readonly description = 'Checks TypeScript source files for circular dependencies';
    public readonly usage = 'code:check-cycles [--path=<relative-path>]';
    public readonly examples = [
        'npm run cli -- code:check-cycles',
        'npm run cli -- code:check-cycles --path=src/module-auth',
    ];

    public run(context: CliCommandContext): CliCommandResult {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            for (const key of Object.keys(context.options)) {
                if (!allowedOptions.has(key)) {
                    context.logger.error(`Unsupported option for code:check-cycles: --${key}`);
                    context.logger.info(`Usage: ${this.usage}`);
                    return { exitCode: 1 };
                }
            }

            const invalidPathResult = rejectBooleanOption(context, 'path', this.usage);
            if (invalidPathResult) {
                return invalidPathResult;
            }

            const pathOption = context.options.path;
            const targetPath = typeof pathOption === 'string' ? pathOption : 'src';
            const projectRoot = resolveCliProjectRoot(__dirname);
            const exitCode = checkCircularDependencies({
                projectRoot,
                targetPath,
                env: { ...process.env, ...context.env },
            });

            if (exitCode !== 0) {
                return {
                    exitCode,
                    error: new Error(`Circular dependency check failed with exit code ${exitCode}`),
                };
            }

            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Circular dependency check failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
