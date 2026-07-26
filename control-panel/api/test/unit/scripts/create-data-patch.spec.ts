import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import {
    createDataPatchFile,
    createRandomPatchSlug,
    createTimestamp,
    parsePatchSlug,
    renderDataPatchTemplate,
    toKebabCase,
} from '../../../scripts/cli/operations/create-data-patch';

describe('create data patch script', () => {
    it('normalizes user input to kebab-case', () => {
        expect(toKebabCase('Patch Name')).toBe('patch-name');
        expect(toKebabCase('identityBootstrap')).toBe('identity-bootstrap');
        expect(toKebabCase('  identity_bootstrap!!  ')).toBe('identity-bootstrap');
    });

    it('creates timestamp names in data patch format', () => {
        expect(createTimestamp(new Date(2026, 6, 3, 0, 1))).toBe('202607030001');
    });

    it('uses provided patch name or random drizzle-style slug', () => {
        expect(parsePatchSlug(['patch-name'])).toBe('patch-name');
        expect(parsePatchSlug(['--name', 'Patch Name'])).toBe('patch-name');
        expect(parsePatchSlug(['--name=Patch Name'])).toBe('patch-name');
        expect(createRandomPatchSlug()).toMatch(/^[a-z]+-[a-z]+$/);
    });

    it('rejects a name flag without a value', () => {
        expect(() => parsePatchSlug(['--name'])).toThrow('Missing value for --name.');
    });

    it('renders a transaction-safe patch template', () => {
        const template = renderDataPatchTemplate('202607030001-patch-name', 'patch name');

        expect(template).toContain("name: '202607030001-patch-name'");
        expect(template).toContain('const { client, logger } = context;');
        expect(template).toContain('await client.query(`');
        expect(template).toContain(
            'logger.error(\'Data patch "202607030001-patch-name" failed.\');',
        );
        expect(template).toContain('throw error;');
        expect(template).toContain('../src/module-data-patch/types/data-patch.types');
    });

    it('creates a data patch file in the target project root', () => {
        const rootDir = mkdtempSync(join(tmpdir(), 'data-patch-generator-'));

        try {
            const filePath = createDataPatchFile(['Patch Name'], rootDir);
            const fileName = basename(filePath);
            const content = readFileSync(filePath, 'utf8');

            expect(fileName).toMatch(/^\d{12}-patch-name\.ts$/);
            expect(content).toContain("description: 'patch name'");
        } finally {
            rmSync(rootDir, { recursive: true, force: true });
        }
    });
});
