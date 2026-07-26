import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

describe('admin bot config UI boundaries', () => {
    it('does not hardcode concrete bot module ids in source files', () => {
        const source = readAdminBotsSource();
        const forbidden = ['spot_grid', 'spot_greenwich', 'spot_grid_bot', 'spot_greenwich_bot'];

        expect(forbidden.filter(phrase => source.includes(phrase))).toEqual([]);
    });

    it('does not make a raw JSON editor the primary workflow', () => {
        const source = readAdminBotsSource();

        expect(source).not.toContain('Raw JSON');
        expect(source).not.toContain('textarea');
    });
});

function readAdminBotsSource(): string {
    const root = process.cwd();
    const files = [
        'src/features/module-admin-bots/components/generic-bot-config-form.tsx',
        'src/features/module-admin-bots/components/admin-bot-config-page.tsx',
        'src/features/module-admin-bots/api-service/client/index.ts',
        'src/features/module-admin-bots/hooks/use-admin-bots.hooks.ts',
    ];

    return files.map(file => readFileSync(join(root, file), 'utf8')).join('\n');
}
