import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const RANDOM_ADJECTIVES = [
    'brave',
    'calm',
    'clear',
    'fresh',
    'gentle',
    'golden',
    'honest',
    'kind',
    'lucky',
    'steady',
    'swift',
    'wise',
];

const RANDOM_NOUNS = [
    'anchor',
    'bridge',
    'harbor',
    'lantern',
    'meadow',
    'orbit',
    'river',
    'signal',
    'summit',
    'voyage',
    'window',
    'zephyr',
];

export function toKebabCase(input: string): string {
    return input
        .trim()
        .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .replace(/-{2,}/g, '-');
}

function pickRandom(values: string[]): string {
    return values[Math.floor(Math.random() * values.length)];
}

export function createRandomPatchSlug(): string {
    return `${pickRandom(RANDOM_ADJECTIVES)}-${pickRandom(RANDOM_NOUNS)}`;
}

export function createTimestamp(date = new Date()): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    return `${year}${month}${day}${hours}${minutes}`;
}

export function parsePatchSlug(argv: string[]): string {
    const nameEqualsArg = argv.find(arg => arg.startsWith('--name='));
    const nameFromEqualsArg = nameEqualsArg?.slice('--name='.length);
    const nameFlagIndex = argv.findIndex(arg => arg === '--name' || arg === '-n');

    if (
        nameFlagIndex >= 0 &&
        (!argv[nameFlagIndex + 1] || argv[nameFlagIndex + 1].startsWith('-'))
    ) {
        throw new Error('Missing value for --name.');
    }

    const rawName =
        nameFromEqualsArg ??
        (nameFlagIndex >= 0 ? argv[nameFlagIndex + 1] : argv.find(arg => !arg.startsWith('-')));
    const slug = toKebabCase(rawName || createRandomPatchSlug());

    if (!slug) {
        throw new Error('Data patch name must contain at least one latin letter or digit.');
    }

    return slug;
}

export function renderDataPatchTemplate(patchName: string, description: string): string {
    return `import type { DataPatch } from '../src/module-data-patch/types/data-patch.types';

export const patch: DataPatch = {
    name: '${patchName}',
    description: '${description}',
    async apply(context): Promise<void> {
        const { client, logger } = context;

        try {
            await client.query(\`
                -- TODO: Implement idempotent data changes.
                -- Example:
                -- insert into "table_name" ("column_name")
                -- values ($1)
                -- on conflict ("column_name") do nothing;
            \`);
        } catch (error) {
            logger.error('Data patch "${patchName}" failed.');
            throw error;
        }
    },
};
`;
}

function getProjectRoot(): string {
    return __dirname.includes('dist')
        ? resolve(__dirname, '../../../..')
        : resolve(__dirname, '../../..');
}

export function createDataPatchFile(argv: string[], rootDir = getProjectRoot()): string {
    const slug = parsePatchSlug(argv);
    const patchName = `${createTimestamp()}-${slug}`;
    const dataPatchesDir = join(rootDir, 'data-patches');
    const filePath = join(dataPatchesDir, `${patchName}.ts`);

    mkdirSync(dataPatchesDir, { recursive: true });

    if (existsSync(filePath)) {
        throw new Error(`Data patch already exists: ${filePath}`);
    }

    const description = slug.replace(/-/g, ' ');
    writeFileSync(filePath, renderDataPatchTemplate(patchName, description), {
        encoding: 'utf8',
        flag: 'wx',
    });

    return filePath;
}
