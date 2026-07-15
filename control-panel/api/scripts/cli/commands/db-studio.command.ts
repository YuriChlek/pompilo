import { spawn } from 'node:child_process';
import path from 'node:path';
import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DbStudioCommand implements CliCommand {
    public readonly name = 'db:studio';
    public readonly description = 'Starts Drizzle Studio to inspect database';
    public readonly usage = 'db:studio';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        return new Promise<CliCommandResult>(resolvePromise => {
            try {
                const unexpectedArgsResult = rejectUnexpectedArgs(context);
                if (unexpectedArgsResult) {
                    resolvePromise(unexpectedArgsResult);
                    return;
                }

                const projectRoot = __dirname.includes('dist')
                    ? path.resolve(__dirname, '../../../..')
                    : path.resolve(__dirname, '../../..');

                context.logger.info('Starting Drizzle Studio...');
                const child = spawn('npx', ['drizzle-kit', 'studio'], {
                    cwd: projectRoot,
                    stdio: 'inherit',
                    env: { ...process.env, ...context.env },
                });

                const sigIntHandler = () => {
                    if (child.pid && !child.killed) {
                        child.kill('SIGINT');
                    }
                };
                const sigTermHandler = () => {
                    if (child.pid && !child.killed) {
                        child.kill('SIGTERM');
                    }
                };

                process.on('SIGINT', sigIntHandler);
                process.on('SIGTERM', sigTermHandler);

                child.on('close', code => {
                    process.off('SIGINT', sigIntHandler);
                    process.off('SIGTERM', sigTermHandler);
                    resolvePromise({ exitCode: code ?? 0 });
                });

                child.on('error', err => {
                    process.off('SIGINT', sigIntHandler);
                    process.off('SIGTERM', sigTermHandler);
                    context.logger.error(`Failed to start Drizzle Studio: ${err.message}`);
                    resolvePromise({ exitCode: 1, error: err });
                });
            } catch (error: unknown) {
                context.logger.error(`Drizzle Studio run failed: ${getErrorMessage(error)}`);
                resolvePromise({ exitCode: 1, error: toError(error) });
            }
        });
    }
}
