import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

type PackageJson = {
    scripts: Record<string, string>;
};

function isPackageJson(value: unknown): value is PackageJson {
    if (!value || typeof value !== 'object') {
        return false;
    }

    const candidate = value as { scripts?: unknown };
    return (
        !!candidate.scripts &&
        typeof candidate.scripts === 'object' &&
        Object.values(candidate.scripts).every(script => typeof script === 'string')
    );
}

function readPackageJson(): PackageJson {
    const packageJsonPath = resolve(__dirname, '../../../package.json');
    const parsed = JSON.parse(readFileSync(packageJsonPath, 'utf8')) as unknown;

    if (!isPackageJson(parsed)) {
        throw new Error('api/package.json must contain a scripts object.');
    }

    return parsed;
}

const permanentScripts = new Set([
    'build',
    'build:cli',
    'cli',
    'format',
    'lint',
    'start',
    'start:dev',
    'start:debug',
    'start:prod',
    'test',
    'test:watch',
    'test:coverage',
    'test:debug',
    'test:unit',
]);

describe('package scripts CLI wrapper policy', () => {
    it('keeps package scripts limited to permanent scripts', () => {
        const { scripts } = readPackageJson();
        const allowedScripts = new Set(permanentScripts);

        const unexpectedScripts = Object.keys(scripts).filter(
            script => !allowedScripts.has(script),
        );

        expect(unexpectedScripts).toEqual([]);
    });

    it('does not keep legacy operational npm wrappers', () => {
        const { scripts } = readPackageJson();
        const removedWrapperScripts = [
            'data-patches:build',
            'data-patches:push',
            'data-patches:create',
            'db:migration:generate',
            'db:migration:run',
            'db:dev:migration:run',
            'db:studio',
            'db:destructive-reset',
            'demo-data:push',
            'create-admin-user',
        ];

        for (const scriptName of removedWrapperScripts) {
            expect(scripts).not.toHaveProperty(scriptName);
        }
    });

    it('keeps build:cli as a source-only bootstrap path independent from compiled CLI', () => {
        const { scripts } = readPackageJson();

        expect(scripts['build:cli']).toBe(
            'node -r ts-node/register -r tsconfig-paths/register ./scripts/build-cli.ts',
        );
        expect(scripts['build:cli']).not.toContain('npm run cli');
        expect(scripts['build:cli']).not.toContain('data-patches:build');
    });

    it('does not use the data-patches:build wrapper as a Docker build bootstrap path', () => {
        const dockerfile = readFileSync(resolve(__dirname, '../../../Dockerfile'), 'utf8');

        expect(dockerfile).toContain('RUN npm run build:cli');
        expect(dockerfile).not.toContain('data-patches:build');
        expect(dockerfile).not.toContain('build:data-patches');
    });
});
