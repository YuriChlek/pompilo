import type { NextRequest } from 'next/server';
import { handleAuthProxy } from '@/features/module-auth/server/handle-auth-proxy';

export function proxy(request: NextRequest) {
    return handleAuthProxy(request);
}

export const config = {
    matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
