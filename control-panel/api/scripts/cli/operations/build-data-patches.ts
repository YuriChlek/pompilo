import { execSync } from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

function calculateChecksum(filePath: string): string {
    const fileBuffer = fs.readFileSync(filePath);
    const hashSum = crypto.createHash('sha256');
    hashSum.update(fileBuffer);
    return `sha256:${hashSum.digest('hex')}`;
}

function collectJavaScriptFiles(directory: string): string[] {
    if (!fs.existsSync(directory)) {
        return [];
    }

    const entries = fs.readdirSync(directory, { withFileTypes: true });
    const files: string[] = [];

    for (const entry of entries) {
        const entryPath = path.join(directory, entry.name);

        if (entry.isDirectory()) {
            files.push(...collectJavaScriptFiles(entryPath));
            continue;
        }

        if (entry.isFile() && entry.name.endsWith('.js')) {
            files.push(entryPath);
        }
    }

    return files;
}

function assertNoUnresolvedAliases(files: string[]): void {
    const offenders = files.filter(file => {
        const content = fs.readFileSync(file, 'utf8');

        return /require\(["']@\//.test(content) || /from ["']@\//.test(content);
    });

    if (offenders.length > 0) {
        let msg = 'Compiled data patch artifacts contain unresolved @/ imports:\n';
        for (const offender of offenders) {
            msg += `- ${offender}\n`;
        }
        throw new Error(msg);
    }
}

export function buildDataPatches(): void {
    const projectRoot = __dirname.includes('dist')
        ? path.resolve(__dirname, '../../../..')
        : path.resolve(__dirname, '../../..');
    const dataPatchesDir = path.join(projectRoot, 'data-patches');
    const distDir = path.join(projectRoot, 'dist');
    const manifestPath = path.join(distDir, 'data-patches-manifest.json');

    console.log('Compiling data patches and scripts...');
    execSync('npx tsc -p tsconfig.data-patches.json', { cwd: projectRoot, stdio: 'inherit' });

    console.log('Generating data patches manifest...');
    if (!fs.existsSync(dataPatchesDir)) {
        throw new Error(`Data patches directory not found at ${dataPatchesDir}`);
    }

    interface PatchManifestEntry {
        name: string;
        sourcePath: string;
        compiledPath: string;
        sourceChecksum: string;
    }

    const files = fs.readdirSync(dataPatchesDir);
    const patches: PatchManifestEntry[] = [];

    // Naming convention regex: YYYYMMDDHHMM-short-kebab-description.ts
    const patchNameRegex = /^\d{12}-[a-z0-9-]+$/;

    for (const file of files) {
        if (!file.endsWith('.ts') || file.endsWith('.spec.ts')) {
            continue;
        }

        const baseName = path.basename(file, '.ts');
        if (!patchNameRegex.test(baseName)) {
            throw new Error(
                `Error: File name "${file}" does not match naming convention YYYYMMDDHHMM-short-kebab-description.ts`,
            );
        }

        const sourceFilePath = path.join(dataPatchesDir, file);
        const checksum = calculateChecksum(sourceFilePath);

        patches.push({
            name: baseName,
            sourcePath: `data-patches/${file}`,
            compiledPath: `data-patches/${baseName}.js`,
            sourceChecksum: checksum,
        });
    }

    // Sort by name ascending to ensure predictable run order
    patches.sort((a, b) => a.name.localeCompare(b.name));

    const manifest = { patches };
    if (!fs.existsSync(distDir)) {
        fs.mkdirSync(distDir, { recursive: true });
    }
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 4));
    console.log(`Successfully generated manifest at ${manifestPath}`);

    assertNoUnresolvedAliases([
        path.join(distDir, 'scripts', 'cli', 'operations', 'run-data-patches.js'),
        ...collectJavaScriptFiles(path.join(distDir, 'data-patches')),
    ]);
    console.log('Compiled data patch artifacts do not contain unresolved @/ imports.');
}
