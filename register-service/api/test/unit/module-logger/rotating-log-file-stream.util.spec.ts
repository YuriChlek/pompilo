import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { RotatingLogFileStream } from '@/module-logger/utils/rotating-log-file-stream.util';

describe('RotatingLogFileStream', () => {
    let directory: string;
    let currentDate: Date;

    beforeEach(() => {
        directory = mkdtempSync(join(tmpdir(), 'pampilo-logs-'));
        currentDate = new Date('2026-07-07T12:00:00.000Z');
    });

    afterEach(() => {
        rmSync(directory, { recursive: true, force: true });
    });

    it('writes logs to a service-scoped daily file', async () => {
        const stream = new RotatingLogFileStream({
            directory,
            maxDays: 14,
            serviceName: 'Pampilo API',
            now: () => currentDate,
        });

        await writeToStream(stream, '{"level":"info"}\n');
        await endStream(stream);

        const filePath = join(directory, 'pampilo-api-2026-07-07.log');

        expect(readFileSync(filePath, 'utf8')).toBe('{"level":"info"}\n');
    });

    it('rotates files when the UTC day changes', async () => {
        const stream = new RotatingLogFileStream({
            directory,
            maxDays: 14,
            serviceName: 'pampilo-api',
            now: () => currentDate,
        });

        await writeToStream(stream, 'before-midnight\n');
        currentDate = new Date('2026-07-08T00:00:01.000Z');
        await writeToStream(stream, 'after-midnight\n');
        await endStream(stream);

        expect(readFileSync(join(directory, 'pampilo-api-2026-07-07.log'), 'utf8')).toBe(
            'before-midnight\n',
        );
        expect(readFileSync(join(directory, 'pampilo-api-2026-07-08.log'), 'utf8')).toBe(
            'after-midnight\n',
        );
    });

    it('removes files older than the configured retention window', async () => {
        writeFileSync(join(directory, 'pampilo-api-2026-07-04.log'), 'expired\n');
        writeFileSync(join(directory, 'pampilo-api-2026-07-06.log'), 'kept\n');

        const stream = new RotatingLogFileStream({
            directory,
            maxDays: 2,
            serviceName: 'pampilo-api',
            now: () => currentDate,
        });

        await writeToStream(stream, 'today\n');
        await endStream(stream);

        expect(existsSync(join(directory, 'pampilo-api-2026-07-04.log'))).toBe(false);
        expect(existsSync(join(directory, 'pampilo-api-2026-07-06.log'))).toBe(true);
        expect(existsSync(join(directory, 'pampilo-api-2026-07-07.log'))).toBe(true);
    });
});

function writeToStream(stream: RotatingLogFileStream, chunk: string): Promise<void> {
    return new Promise((resolve, reject) => {
        stream.write(chunk, error => {
            if (error) {
                reject(error);
                return;
            }

            resolve();
        });
    });
}

function endStream(stream: RotatingLogFileStream): Promise<void> {
    return new Promise((resolve, reject) => {
        stream.end(error => {
            if (error) {
                reject(error instanceof Error ? error : new Error(String(error)));
                return;
            }

            resolve();
        });
    });
}
