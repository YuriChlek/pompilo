import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { seedDemoData } from '../operations/demo-data/seed-demo-data';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

export class DemoDataPushCommand implements CliCommand {
    public readonly name = 'demo-data:push';
    public readonly description = 'Seeds demo identity users for development';
    public readonly usage = 'demo-data:push';

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            await seedDemoData();
            context.logger.info('Demo data successfully seeded.');
            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Demo data seeding failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
