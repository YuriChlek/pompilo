import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DbDevMigrationRunCommand implements CliCommand {
    public readonly name = 'db:dev:migration:run';
    public readonly description = 'Applies pending Drizzle migrations with NODE_ENV=development';
    public readonly usage = 'db:dev:migration:run';

    public run(context: CliCommandContext): CliCommandResult {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            const projectRoot = __dirname.includes('dist')
                ? path.resolve(__dirname, '../../../..')
                : path.resolve(__dirname, '../../..');

            const result = spawnSync('npx', ['drizzle-kit', 'migrate'], {
                cwd: projectRoot,
                stdio: 'inherit',
                env: { ...process.env, ...context.env, NODE_ENV: 'development' },
            });

            if (result.status !== 0) {
                return {
                    exitCode: result.status ?? 1,
                    error: new Error(
                        `Drizzle kit migrate (dev) failed with exit code ${result.status}`,
                    ),
                };
            }

            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Dev migration run failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
