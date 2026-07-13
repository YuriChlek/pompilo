/* eslint-disable @typescript-eslint/no-unsafe-assignment */

import { runDataPatches, FsOverrides } from '../../../scripts/cli/operations/run-data-patches';
import type { Pool } from 'pg';

interface MockClient {
    query: jest.Mock<Promise<{ rows: Record<string, unknown>[] }>, [string, unknown[]?]>;
    release: jest.Mock<void, []>;
}

interface MockPool {
    connect: jest.Mock<Promise<MockClient>, []>;
    end: jest.Mock<Promise<void>, []>;
}

interface MockPatch {
    name: string;
    description: string;
    apply: jest.Mock<Promise<void>, [unknown]>;
}

describe('runDataPatches script', () => {
    let mockClient: MockClient;
    let mockPool: MockPool;
    let mockFs: FsOverrides;

    const DUMMY_FILE_CHECKSUM =
        'sha256:fc750b6744db7aeb9458ab8800020e0bab8911891fe7c2804f6f2612e8adb967';

    beforeEach(() => {
        process.env.DB_HOST = 'localhost';
        process.env.DB_USER = 'admin';
        process.env.DB_NAME = 'test_db';

        mockClient = {
            query: jest.fn().mockImplementation((queryText: string) => {
                if (queryText.includes('pg_tables')) {
                    return Promise.resolve({ rows: [{ exists: true }] });
                }
                if (queryText.includes('SELECT patch_name')) {
                    return Promise.resolve({ rows: [] });
                }
                return Promise.resolve({ rows: [] });
            }),
            release: jest.fn(),
        };

        mockPool = {
            connect: jest.fn().mockResolvedValue(mockClient),
            end: jest.fn().mockResolvedValue(undefined),
        };

        const mockRequire = jest.fn().mockImplementation((pathStr: string) => {
            const sportsApply = jest.fn().mockResolvedValue(undefined);
            const userApply = jest.fn().mockResolvedValue(undefined);

            if (pathStr.includes('202607030001')) {
                const patch: MockPatch = {
                    name: '202607030001-identity-bootstrap',
                    description: 'sports',
                    apply: sportsApply,
                };
                return { patch };
            }
            if (pathStr.includes('202607030002')) {
                const patch: MockPatch = {
                    name: '202607030002-backfill-user-settings',
                    description: 'users',
                    apply: userApply,
                };
                return { patch };
            }
            return {};
        });

        mockFs = {
            existsSync: jest.fn().mockReturnValue(true),
            readFileSync: jest.fn().mockImplementation((pathStr: string) => {
                if (pathStr.includes('.env')) {
                    return 'DB_HOST=localhost\nDB_USER=admin\nDB_NAME=test_db';
                }
                return 'dummy ts file content';
            }),
            readdirSync: jest
                .fn()
                .mockReturnValue([
                    '202607030001-identity-bootstrap.ts',
                    '202607030002-backfill-user-settings.ts',
                ]),
            requireOverride: mockRequire,
        };
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('successfully processes all pending patches in alphabetical order', async () => {
        const result = await runDataPatches(
            ['node', 'script'],
            mockPool as unknown as Pool,
            mockFs,
        );

        expect(result).toEqual({ total: 2, skipped: 0, applied: 2 });

        // Verify advisory lock
        expect(mockClient.query).toHaveBeenCalledWith(
            expect.stringContaining("pg_advisory_lock(hashtext('pampilo_data_patches_runner'))"),
        );
        expect(mockClient.query).toHaveBeenCalledWith(
            expect.stringContaining("pg_advisory_unlock(hashtext('pampilo_data_patches_runner'))"),
        );

        // Verify transaction statements
        expect(mockClient.query).toHaveBeenCalledWith('BEGIN');
        expect(mockClient.query).toHaveBeenCalledWith('COMMIT');
        expect(mockClient.query).toHaveBeenCalledWith(
            expect.stringContaining('INSERT INTO data_patches'),
            expect.any(Array),
        );

        // Check release calls
        expect(mockClient.release).toHaveBeenCalledTimes(3); // 1 for lockClient, 2 for patch clients
    });

    it('skips already applied patches', async () => {
        mockClient.query = jest.fn().mockImplementation((queryText: string) => {
            if (queryText.includes('pg_tables')) {
                return Promise.resolve({ rows: [{ exists: true }] });
            }
            if (queryText.includes('SELECT patch_name')) {
                return Promise.resolve({
                    rows: [
                        {
                            patchName: '202607030001-identity-bootstrap',
                            checksum: DUMMY_FILE_CHECKSUM,
                        },
                    ],
                });
            }
            return Promise.resolve({ rows: [] });
        });

        const result = await runDataPatches(
            ['node', 'script'],
            mockPool as unknown as Pool,
            mockFs,
        );

        expect(result).toEqual({ total: 2, skipped: 1, applied: 1 });
    });

    it('throws error on checksum drift for already applied patch', async () => {
        mockClient.query = jest.fn().mockImplementation((queryText: string) => {
            if (queryText.includes('pg_tables')) {
                return Promise.resolve({ rows: [{ exists: true }] });
            }
            if (queryText.includes('SELECT patch_name')) {
                return Promise.resolve({
                    rows: [
                        {
                            patchName: '202607030001-identity-bootstrap',
                            checksum: 'sha256:mismatched_checksum',
                        },
                    ],
                });
            }
            return Promise.resolve({ rows: [] });
        });

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('Checksum drift detected');
    });

    it('supports dry-run mode and does not modify database', async () => {
        const result = await runDataPatches(
            ['node', 'script', '--dry-run'],
            mockPool as unknown as Pool,
            mockFs,
        );

        expect(result).toEqual({ total: 2, skipped: 0, applied: 0 });

        // Ensure no transaction was started and no row inserted
        expect(mockClient.query).not.toHaveBeenCalledWith('BEGIN');
        expect(mockClient.query).not.toHaveBeenCalledWith(
            expect.stringContaining('INSERT INTO data_patches'),
            expect.any(Array),
        );
    });

    it('throws error if data_patches table is missing', async () => {
        mockClient.query = jest.fn().mockImplementation((queryText: string) => {
            if (queryText.includes('pg_tables')) {
                return Promise.resolve({ rows: [{ exists: false }] });
            }
            return Promise.resolve({ rows: [] });
        });

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('Table "data_patches" does not exist');
    });

    it('throws error if a patch file violates the naming convention', async () => {
        mockFs.readdirSync = jest.fn().mockReturnValue(['invalid-name.ts']);

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('does not match naming convention');
    });

    it('throws error if duplicate patch names exist', async () => {
        mockFs.readdirSync = jest
            .fn()
            .mockReturnValue([
                '202607030001-identity-bootstrap.ts',
                '202607030001-identity-bootstrap.ts',
            ]);

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('Duplicate patch name found');
    });

    it('throws error if patch file does not export a patch object', async () => {
        mockFs.requireOverride = jest.fn().mockReturnValue({});

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('does not export "patch" object');
    });

    it('throws error and performs rollback when a patch fails', async () => {
        const failingPatchApply = jest.fn().mockRejectedValue(new Error('Apply error'));

        mockFs.requireOverride = jest.fn().mockImplementation((pathStr: string) => {
            if (pathStr.includes('202607030001')) {
                const patch: MockPatch = {
                    name: '202607030001-identity-bootstrap',
                    description: 'sports',
                    apply: failingPatchApply,
                };
                return { patch };
            }
            return {};
        });

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('Transaction rolled back. Error: Apply error');

        expect(mockClient.query).toHaveBeenCalledWith('BEGIN');
        expect(mockClient.query).toHaveBeenCalledWith('ROLLBACK');
        expect(mockClient.query).not.toHaveBeenCalledWith('COMMIT');
    });

    it('loads compiled patches from manifest and uses source checksum', async () => {
        const compiledChecksum = 'sha256:compiled-source-checksum';

        mockFs = {
            ...mockFs,
            mode: 'compiled',
            readFileSync: jest.fn().mockImplementation((pathStr: string) => {
                if (pathStr.includes('.env')) {
                    return 'DB_HOST=localhost\nDB_USER=admin\nDB_NAME=test_db';
                }
                if (pathStr.includes('data-patches-manifest.json')) {
                    return JSON.stringify({
                        patches: [
                            {
                                name: '202607030001-identity-bootstrap',
                                sourcePath: 'data-patches/202607030001-identity-bootstrap.ts',
                                compiledPath: 'data-patches/202607030001-identity-bootstrap.js',
                                sourceChecksum: compiledChecksum,
                            },
                        ],
                    });
                }
                return 'unused';
            }),
            readdirSync: jest.fn().mockReturnValue([]),
        };

        const result = await runDataPatches(
            ['node', 'script'],
            mockPool as unknown as Pool,
            mockFs,
        );

        expect(result).toEqual({ total: 1, skipped: 0, applied: 1 });
        expect(mockClient.query).toHaveBeenCalledWith(
            expect.stringContaining('INSERT INTO data_patches'),
            expect.arrayContaining(['202607030001-identity-bootstrap', compiledChecksum]),
        );
        expect(mockFs.requireOverride).toHaveBeenCalledWith(
            expect.stringContaining('data-patches/202607030001-identity-bootstrap.js'),
        );
    });

    it('throws error when compiled manifest is missing patches array', async () => {
        mockFs = {
            ...mockFs,
            mode: 'compiled',
            readFileSync: jest.fn().mockImplementation((pathStr: string) => {
                if (pathStr.includes('.env')) {
                    return 'DB_HOST=localhost\nDB_USER=admin\nDB_NAME=test_db';
                }
                if (pathStr.includes('data-patches-manifest.json')) {
                    return JSON.stringify({});
                }
                return 'unused';
            }),
        };

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('must contain a patches array');
    });

    it('throws error when compiled patch listed in manifest is missing', async () => {
        mockFs = {
            ...mockFs,
            mode: 'compiled',
            existsSync: jest.fn().mockImplementation((pathStr: string) => {
                return !pathStr.includes('202607030001-identity-bootstrap.js');
            }),
            readFileSync: jest.fn().mockImplementation((pathStr: string) => {
                if (pathStr.includes('.env')) {
                    return 'DB_HOST=localhost\nDB_USER=admin\nDB_NAME=test_db';
                }
                if (pathStr.includes('data-patches-manifest.json')) {
                    return JSON.stringify({
                        patches: [
                            {
                                name: '202607030001-identity-bootstrap',
                                sourcePath: 'data-patches/202607030001-identity-bootstrap.ts',
                                compiledPath: 'data-patches/202607030001-identity-bootstrap.js',
                                sourceChecksum: DUMMY_FILE_CHECKSUM,
                            },
                        ],
                    });
                }
                return 'unused';
            }),
        };

        await expect(
            runDataPatches(['node', 'script'], mockPool as unknown as Pool, mockFs),
        ).rejects.toThrow('Compiled data patch file not found');
    });
});
