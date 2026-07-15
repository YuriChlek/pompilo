import {
    createWriteStream,
    existsSync,
    mkdirSync,
    readdirSync,
    rmSync,
    type WriteStream,
} from 'fs';
import { join } from 'path';
import { Writable } from 'stream';

const DAY_MS = 24 * 60 * 60 * 1000;

interface RotatingLogFileStreamOptions {
    directory: string;
    maxDays: number;
    serviceName: string;
    now?: () => Date;
}

export class RotatingLogFileStream extends Writable {
    private readonly now: () => Date;
    private readonly filePrefix: string;
    private currentDateKey?: string;
    private currentStream?: WriteStream;

    constructor(private readonly options: RotatingLogFileStreamOptions) {
        super();
        this.now = options.now ?? (() => new Date());
        this.filePrefix = sanitizeLogFilePrefix(options.serviceName);
        mkdirSync(options.directory, { recursive: true });
        this.cleanupOldFiles();
    }

    getCurrentFilePath(): string {
        return this.getFilePath(this.getDateKey(this.now()));
    }

    override _write(
        chunk: Buffer | string,
        encoding: BufferEncoding,
        callback: (error?: Error | null) => void,
    ): void {
        try {
            const stream = this.getStream();
            stream.write(chunk, encoding, callback);
        } catch (error) {
            callback(error instanceof Error ? error : new Error(String(error)));
        }
    }

    override _final(callback: (error?: Error | null) => void): void {
        if (!this.currentStream) {
            callback();
            return;
        }

        this.currentStream.end(callback);
    }

    private getStream(): WriteStream {
        const dateKey = this.getDateKey(this.now());

        if (this.currentStream && this.currentDateKey === dateKey) {
            return this.currentStream;
        }

        if (this.currentStream) {
            this.currentStream.end();
        }

        this.currentDateKey = dateKey;
        this.currentStream = createWriteStream(this.getFilePath(dateKey), { flags: 'a' });
        this.cleanupOldFiles();

        return this.currentStream;
    }

    private cleanupOldFiles(): void {
        if (!existsSync(this.options.directory)) {
            return;
        }

        const cutoffTime =
            startOfUtcDay(this.now()).getTime() - (this.options.maxDays - 1) * DAY_MS;
        const filePattern = new RegExp(
            `^${escapeRegExp(this.filePrefix)}-(\\d{4}-\\d{2}-\\d{2})\\.log$`,
        );

        for (const fileName of readdirSync(this.options.directory)) {
            const match = filePattern.exec(fileName);

            if (!match) {
                continue;
            }

            const fileDate = Date.parse(`${match[1]}T00:00:00.000Z`);

            if (Number.isFinite(fileDate) && fileDate < cutoffTime) {
                rmSync(join(this.options.directory, fileName), { force: true });
            }
        }
    }

    private getFilePath(dateKey: string): string {
        return join(this.options.directory, `${this.filePrefix}-${dateKey}.log`);
    }

    private getDateKey(date: Date): string {
        return date.toISOString().slice(0, 10);
    }
}

export function createRotatingLogFileStream(
    options: RotatingLogFileStreamOptions,
): RotatingLogFileStream {
    return new RotatingLogFileStream(options);
}

function sanitizeLogFilePrefix(serviceName: string): string {
    const sanitized = serviceName
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9._-]+/g, '-')
        .replace(/^-+|-+$/g, '');

    return sanitized.length > 0 ? sanitized : 'app';
}

function startOfUtcDay(date: Date): Date {
    return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
