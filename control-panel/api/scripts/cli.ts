import { CommandRegistry } from './cli/command-registry';
import { HelpCommand } from './cli/commands/help.command';
import { DataPatchesPushCommand } from './cli/commands/data-patches-push.command';
import { DataPatchesListCommand } from './cli/commands/data-patches-list.command';
import { DataPatchesDryRunCommand } from './cli/commands/data-patches-dry-run.command';
import { DataPatchesCreateCommand } from './cli/commands/data-patches-create.command';
import { DataPatchesBuildCommand } from './cli/commands/data-patches-build.command';
import { DbMigrationGenerateCommand } from './cli/commands/db-migration-generate.command';
import { DbMigrationRunCommand } from './cli/commands/db-migration-run.command';
import { DbDevMigrationRunCommand } from './cli/commands/db-dev-migration-run.command';
import { DbStudioCommand } from './cli/commands/db-studio.command';
import { DbDestructiveResetCommand } from './cli/commands/db-destructive-reset.command';
import { AdminCreateCommand } from './cli/commands/admin-create.command';
import { DemoDataPushCommand } from './cli/commands/demo-data-push.command';
import { CodeCheckCyclesCommand } from './cli/commands/code-check-cycles.command';
import { ConsoleLogger } from './cli/cli-logger';
import { parseCliArgv } from './cli/parse-cli-argv';
import { getErrorMessage } from './cli/error-format';

export function createBuiltInCommandRegistry(): CommandRegistry {
    const registry = new CommandRegistry();

    registry.register(new HelpCommand(registry));
    registry.register(new DataPatchesPushCommand());
    registry.register(new DataPatchesListCommand());
    registry.register(new DataPatchesDryRunCommand());
    registry.register(new DataPatchesCreateCommand());
    registry.register(new DataPatchesBuildCommand());
    registry.register(new DbMigrationGenerateCommand());
    registry.register(new DbMigrationRunCommand());
    registry.register(new DbDevMigrationRunCommand());
    registry.register(new DbStudioCommand());
    registry.register(new DbDestructiveResetCommand());
    registry.register(new AdminCreateCommand());
    registry.register(new DemoDataPushCommand());
    registry.register(new CodeCheckCyclesCommand());

    return registry;
}

export async function runCli(argv: string[], registryOverride?: CommandRegistry): Promise<number> {
    const logger = new ConsoleLogger();

    // 1. Initialize registry and register built-in commands
    const registry = registryOverride || createBuiltInCommandRegistry();

    // 2. Parse argv
    const parsed = parseCliArgv(argv);

    // 3. Handle empty commandName
    if (!parsed.commandName) {
        logger.error('No command specified.');
        const helpCmd = registry.get('help');
        if (helpCmd) {
            await helpCmd.run({
                commandName: 'help',
                args: [],
                options: {},
                env: process.env,
                cwd: process.cwd(),
                logger,
            });
        }
        return 1;
    }

    // 4. Find command
    const command = registry.get(parsed.commandName);
    if (!command) {
        logger.error(`Unknown command: ${parsed.commandName}`);
        logger.info('Use "npm run cli -- help" to see all available commands.');
        return 1;
    }

    // 5. Execute command
    try {
        const context = {
            commandName: parsed.commandName,
            args: parsed.args,
            options: parsed.options,
            env: process.env,
            cwd: process.cwd(),
            logger,
        };

        const result = await command.run(context);
        if (result.error) {
            logger.error(`Command failed with error: ${result.error.message}`);
        }
        return result.exitCode;
    } catch (err: unknown) {
        logger.error(`Unhandled error during command execution: ${getErrorMessage(err)}`);
        return 1;
    }
}

// Runnable entrypoint check
const isMain =
    require.main === module || (process.argv[1] && require.resolve(process.argv[1]) === __filename);

if (isMain) {
    runCli(process.argv.slice(2))
        .then(exitCode => {
            process.exitCode = exitCode;
        })
        .catch(err => {
            console.error('Fatal CLI Error:', err);
            process.exitCode = 1;
        });
}
