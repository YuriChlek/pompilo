import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
    test: {
        environment: 'jsdom',
        globals: true,
        setupFiles: ['./src/tests/setup.ts'],
        include: ['./src/tests/**/*.test.ts', './src/tests/**/*.test.tsx'],
        css: true,
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            '@public': path.resolve(__dirname, './public'),
            'server-only': path.resolve(__dirname, './src/tests/mocks/server-only.ts'),
        },
    },
});
