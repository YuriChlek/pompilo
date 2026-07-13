import 'server-only';

import { cookies, headers } from 'next/headers';
import { Dispatcher } from 'undici-types';
import { HttpResponse, RequestConfig } from '@/lib/http-client/interfaces/http-client.interfaces';
import { HttpData } from '@/lib/http-client/types/http-client.types';
import { httpClientConfig } from '@/lib/http-client/config/http-client.config';
import { AbstractHttpClient } from '@/lib/http-client/abstract-http-client';
import {
    buildAccessCookieHeader,
    buildRefreshCookieHeader,
    getCookieHeaderFromSetCookieHeaders,
    getSetCookieHeaders,
    mergeCookieHeaders,
} from '@/features/module-auth/server/auth-cookie-header';
import { getAuthScopeFromPath } from '@/features/module-auth/lib/auth-refresh';
import { AUTH_SCOPE_CONFIG } from '@/features/module-auth/config/auth-scope.config';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

function resolveNextOptions(cache: RequestCache | undefined, next: RequestConfig['next']) {
    return cache === 'no-store' ? undefined : next;
}

class ServerHttpClient extends AbstractHttpClient {
    private readonly requestAuthScopes = new WeakMap<RequestInit, AuthScope>();
    private readonly refreshedCookies = new Map<string, string>();

    protected async request<TResponse, TBody = undefined>(
        path: string,
        method: Dispatcher.HttpMethod,
        body?: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>> {
        const url = new URL(path, this.baseUrl);

        if (config?.params) {
            Object.entries(config.params).forEach(([key, value]) => {
                if (value !== undefined) {
                    url.searchParams.append(key, String(value));
                }
            });
        }

        const scope = config?.authScope ?? getAuthScopeFromPath(path);
        const headersInit = new Headers({ ...this.defaultHeaders });
        const cookieStore = await cookies();
        const cookieHeader =
            this.buildRefreshedAccessCookieHeader(scope) || buildAccessCookieHeader(cookieStore, scope);

        if (cookieHeader) {
            headersInit.set('cookie', cookieHeader);
        }
        headersInit.set('user-agent', (await headers()).get('user-agent') ?? '');

        const cache = config?.cache ?? this.cache;
        const options: RequestInit = {
            method,
            headers: headersInit,
            body: body ? JSON.stringify(body) : undefined,
            credentials: 'include',
            cache,
            next: resolveNextOptions(cache, config?.next ?? this.next),
            signal: config?.signal,
        };

        if (config?.authScope) {
            this.requestAuthScopes.set(options, config.authScope);
        }

        return this.fetchData(options, url, path);
    }

    protected async getRefreshOptions(path: string, options?: RequestInit): Promise<RequestInit> {
        const scope = this.getAuthScope(path, options);
        const cookieStore = await cookies();
        const refreshCookieHeader =
            this.buildRefreshedRefreshCookieHeader(scope) ||
            buildRefreshCookieHeader(cookieStore, scope);
        const deviceId =
            this.refreshedCookies.get(COOKIE_NAMES.DEVICE_ID) ??
            cookieStore.get(COOKIE_NAMES.DEVICE_ID)?.value;
        const headersInit = new Headers();
        const requestHeaders = await headers();

        let cookieHeader = refreshCookieHeader;
        if (deviceId) {
            cookieHeader = cookieHeader
                ? `${cookieHeader}; ${COOKIE_NAMES.DEVICE_ID}=${deviceId}`
                : `${COOKIE_NAMES.DEVICE_ID}=${deviceId}`;
        }

        if (cookieHeader) {
            headersInit.set('cookie', cookieHeader);
        }

        headersInit.set('user-agent', requestHeaders.get('user-agent') ?? '');
        headersInit.set('x-forwarded-for', requestHeaders.get('x-forwarded-for') ?? '');
        headersInit.set('x-real-ip', requestHeaders.get('x-real-ip') ?? '');

        return {
            method: 'POST',
            headers: headersInit,
            cache: 'no-store',
        };
    }

    protected getRetryOptionsAfterRefresh(
        path: string,
        options: RequestInit,
        response: Response,
    ): RequestInit {
        const scope = this.getAuthScope(path, options);
        const accessCookieName = AUTH_SCOPE_CONFIG[scope].accessCookie;
        const accessCookie = getCookieHeaderFromSetCookieHeaders(
            response.headers,
            accessCookieName,
        );

        this.storeSetCookieHeaders(response.headers);

        if (!accessCookie) {
            return options;
        }

        const headersInit = new Headers(options.headers);
        headersInit.set('cookie', mergeCookieHeaders(headersInit.get('cookie'), accessCookie));

        return {
            ...options,
            headers: headersInit,
        };
    }

    private getAuthScope(path: string, options?: RequestInit): AuthScope {
        return (
            (options ? this.requestAuthScopes.get(options) : undefined) ??
            getAuthScopeFromPath(path)
        );
    }

    private buildRefreshedAccessCookieHeader(scope: AuthScope): string {
        const accessCookieName = AUTH_SCOPE_CONFIG[scope].accessCookie;
        const accessCookieValue = this.refreshedCookies.get(accessCookieName);

        return accessCookieValue ? `${accessCookieName}=${accessCookieValue}` : '';
    }

    private buildRefreshedRefreshCookieHeader(scope: AuthScope): string {
        const refreshCookieName = AUTH_SCOPE_CONFIG[scope].refreshCookie;
        const refreshCookieValue = this.refreshedCookies.get(refreshCookieName);

        return refreshCookieValue ? `${refreshCookieName}=${refreshCookieValue}` : '';
    }

    private storeSetCookieHeaders(headers: Headers): void {
        for (const setCookie of getSetCookieHeaders(headers)) {
            const [nameValue = ''] = setCookie.split(';');
            const trimmedNameValue = nameValue.trim();

            if (!trimmedNameValue) {
                continue;
            }

            const [name, ...valueParts] = trimmedNameValue.split('=');

            if (name) {
                this.refreshedCookies.set(name, valueParts.join('='));
            }
        }
    }
}

export const getApiServerClient = async () => {
    return new ServerHttpClient(httpClientConfig);
};
