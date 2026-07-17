import { CliCommand, CliCommandContext, CliCommandResult } from '../cli-command.types';
import { createAdminUser } from '../operations/create-admin-user';
import { rejectUnexpectedArgs } from '../command-validation';
import { getErrorMessage, toError } from '../error-format';

const allowedOptions = new Set([
    'admin-email',
    'admin-password',
    'admin-firstname',
    'admin-lastname',
    'role',
]);

export class AdminCreateCommand implements CliCommand {
    public readonly name = 'admin:create';
    public readonly description = 'Creates a new admin or superAdmin user';
    public readonly usage =
        'admin:create --admin-email=<email> --admin-password=<password> --admin-firstname=<firstname> --admin-lastname=<lastname> [--role=<role>]';
    public readonly examples = [
        'npm run cli -- admin:create --admin-email=admin@example.com --admin-password=SecurePassword123 --admin-firstname=John --admin-lastname=Doe --role=platformAdmin',
    ];

    public async run(context: CliCommandContext): Promise<CliCommandResult> {
        try {
            const unexpectedArgsResult = rejectUnexpectedArgs(context);
            if (unexpectedArgsResult) {
                return unexpectedArgsResult;
            }

            for (const [key, value] of Object.entries(context.options)) {
                if (!allowedOptions.has(key)) {
                    context.logger.error(`Unsupported option for admin:create: --${key}`);
                    context.logger.info(`Usage: ${this.usage}`);
                    return { exitCode: 1 };
                }
                if (value === true) {
                    context.logger.error(`Option --${key} must use --${key}=value format.`);
                    context.logger.info(`Usage: ${this.usage}`);
                    return { exitCode: 1 };
                }
            }

            for (const requiredOption of [
                'admin-email',
                'admin-password',
                'admin-firstname',
                'admin-lastname',
            ]) {
                if (typeof context.options[requiredOption] !== 'string') {
                    context.logger.error(`Missing required option: --${requiredOption}=<value>`);
                    context.logger.info(`Usage: ${this.usage}`);
                    return { exitCode: 1 };
                }
            }

            const stringOptions: Record<string, string> = {};
            for (const [key, value] of Object.entries(context.options)) {
                stringOptions[key] = String(value);
            }

            const result = await createAdminUser(stringOptions);
            context.logger.info(`Admin user successfully created.`);
            context.logger.info(`ID:       ${result.userId}`);
            context.logger.info(`Name:     ${result.name}`);
            context.logger.info(`Email:    ${result.email}`);
            context.logger.info(`Role:     ${result.role}`);

            return { exitCode: 0 };
        } catch (error: unknown) {
            context.logger.error(`Admin user creation failed: ${getErrorMessage(error)}`);
            return { exitCode: 1, error: toError(error) };
        }
    }
}
