import { CliLogger } from './cli-command.types';

export class ConsoleLogger implements CliLogger {
    public info(message: string): void {
        console.log(message);
    }

    public warn(message: string): void {
        console.warn(`\x1b[33m[WARNING] ${message}\x1b[0m`);
    }

    public error(message: string): void {
        console.error(`\x1b[31m[ERROR] ${message}\x1b[0m`);
    }
}
