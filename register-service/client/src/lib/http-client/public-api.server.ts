import 'server-only';

import { apiBaseUrl } from '@/lib/config/api-base-url.config';

interface PublicApiResponse<T> {
    data?: T;
    message?: string;
}

interface PublicApiRequestOptions {
    params?: Record<string, string | undefined>;
}

export async function getPublicApiData<T>(
    path: string,
    options: PublicApiRequestOptions = {},
): Promise<T> {
    const url = new URL(path, apiBaseUrl);

    for (const [key, value] of Object.entries(options.params ?? {})) {
        if (value !== undefined) {
            url.searchParams.set(key, value);
        }
    }

    const response = await fetch(url, {
        headers: {
            Accept: 'application/json',
        },
        cache: 'no-store',
    });
    const payload = (await response.json().catch(() => ({}))) as PublicApiResponse<T>;

    if (!response.ok || payload.data === undefined) {
        throw new Error(payload.message ?? `Public API request failed with status ${response.status}`);
    }

    return payload.data;
}
