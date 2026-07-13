import { apiBaseUrl } from '@/lib/config/api-base-url.config';
import { filterApiProxyCookieHeader, getSetCookieHeaders } from '@/features/module-auth/server/auth-cookie-header';

type RequestParams = { params: Promise<{ slug: string[] }> };
const HOP_BY_HOP_HEADERS = new Set([
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
]);

async function proxyRequest(request: Request, paramsPromise: RequestParams['params']) {
    if (!apiBaseUrl) {
        return new Response('API_BASE_URL environment variable is not configured', { status: 500 });
    }

    const { slug } = await paramsPromise;
    const pathname = slug.join('/');
    const proxyURL = new URL(pathname, apiBaseUrl);
    const requestURL = new URL(request.url);
    const proxyHeaders = new Headers(request.headers);

    proxyURL.search = requestURL.search;

    for (const header of HOP_BY_HOP_HEADERS) {
        proxyHeaders.delete(header);
    }

    const filteredCookieHeader = filterApiProxyCookieHeader(proxyHeaders.get('cookie'), pathname);

    if (filteredCookieHeader) {
        proxyHeaders.set('cookie', filteredCookieHeader);
    } else {
        proxyHeaders.delete('cookie');
    }

    const proxyRequest = new Request(proxyURL, {
        method: request.method,
        headers: proxyHeaders,
        body: request.body,
        duplex: 'half',
    } as RequestInit);

    try {
        const response = await fetch(proxyRequest);

        const responseHeaders = new Headers();
        for (const [key, value] of response.headers.entries()) {
            if (key.toLowerCase() !== 'set-cookie') {
                responseHeaders.set(key, value);
            }
        }

        const setCookies = getSetCookieHeaders(response.headers);
        for (const cookie of setCookies) {
            responseHeaders.append('set-cookie', cookie);
        }

        return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: responseHeaders,
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected exception';

        return new Response(message, { status: 503 });
    }
}

export async function POST(request: Request, { params }: RequestParams) {
    return proxyRequest(request, params);
}

export async function GET(request: Request, { params }: RequestParams) {
    return proxyRequest(request, params);
}

export async function DELETE(request: Request, { params }: RequestParams) {
    return proxyRequest(request, params);
}

export async function PUT(request: Request, { params }: RequestParams) {
    return proxyRequest(request, params);
}

export async function PATCH(request: Request, { params }: RequestParams) {
    return proxyRequest(request, params);
}
