import { spawnSync } from 'node:child_process';
import path from 'node:path';

export type CheckCircularDependenciesOptions = {
    projectRoot: string;
    targetPath: string;
    env: NodeJS.ProcessEnv;
};

export function checkCircularDependencies({
    projectRoot,
    targetPath,
    env,
}: CheckCircularDependenciesOptions): number {
    const result = spawnSync(
        'npx',
        ['madge', '--ts-config', 'tsconfig.json', '--extensions', 'ts', '--circular', targetPath],
        {
            cwd: projectRoot,
            stdio: 'inherit',
            env,
        },
    );

    return result.status ?? 1;
}

export function resolveCliProjectRoot(commandDirname: string): string {
    return commandDirname.includes('dist')
        ? path.resolve(commandDirname, '../../../..')
        : path.resolve(commandDirname, '../../..');
}
