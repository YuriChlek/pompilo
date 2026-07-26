import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

function collectPatchFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
        const entryPath = join(directory, entry.name);

        if (entry.isDirectory()) {
            return collectPatchFiles(entryPath);
        }

        return entry.isFile() && entry.name.endsWith('.ts') && !entry.name.endsWith('.spec.ts')
            ? [entryPath]
            : [];
    });
}

describe('data patch build policy', () => {
    it('does not allow @ aliases in data patch source files', () => {
        const dataPatchesDir = resolve(__dirname, '../../../data-patches');
        const offenders = collectPatchFiles(dataPatchesDir).filter(file => {
            const content = readFileSync(file, 'utf8');

            return /from ['"]@\//.test(content) || /require\(['"]@\//.test(content);
        });

        expect(offenders).toEqual([]);
    });
});
