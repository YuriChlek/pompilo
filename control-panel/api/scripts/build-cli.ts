import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { getErrorMessage } from './cli/error-format';
import { buildDataPatches } from './cli/operations/build-data-patches';

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
        let msg = 'Compiled CLI artifacts contain unresolved @/ imports:\n';
        for (const offender of offenders) {
            msg += `- ${offender}\n`;
        }
        throw new Error(msg);
    }
}

export function buildCli(): void {
    const projectRoot = path.resolve(__dirname, '..');
    const distDir = path.join(projectRoot, 'dist');
    const cliJsPath = path.join(distDir, 'scripts', 'cli.js');

    console.log('Compiling CLI and operational scripts...');
    execSync('npx tsc -p tsconfig.cli.json', { cwd: projectRoot, stdio: 'inherit' });
    buildDataPatches();

    if (!fs.existsSync(cliJsPath)) {
        throw new Error(`Compiled CLI entrypoint not found at ${cliJsPath}`);
    }
    console.log(`Compiled CLI entrypoint found at ${cliJsPath}`);

    console.log('Scanning CLI build output for unresolved @/ aliases...');
    const filesToScan = [
        ...collectJavaScriptFiles(path.join(distDir, 'scripts')),
        ...collectJavaScriptFiles(path.join(distDir, 'data-patches')),
    ];
    assertNoUnresolvedAliases(filesToScan);
    console.log('All CLI build output scanned. No unresolved @/ aliases found.');
}

if (require.main === module) {
    try {
        buildCli();
    } catch (err: unknown) {
        console.error('CLI Build failed:', getErrorMessage(err));
        process.exit(1);
    }
}
