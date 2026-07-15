import { CliCommand } from './cli-command.types';

export class CommandRegistry {
    private readonly commands = new Map<string, CliCommand>();

    public register(command: CliCommand): void {
        const normalizedName = command.name.trim().toLowerCase();
        if (this.commands.has(normalizedName)) {
            throw new Error(`Duplicate command registered: ${command.name}`);
        }
        this.commands.set(normalizedName, command);
    }

    public get(name: string): CliCommand | undefined {
        return this.commands.get(name.trim().toLowerCase());
    }

    public getAll(): CliCommand[] {
        return Array.from(this.commands.values());
    }

    public clear(): void {
        this.commands.clear();
    }
}
