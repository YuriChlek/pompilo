import {
    HttpClientOptions,
    RequestConfig,
    HttpError,
    HttpResponse,
} from '@/lib/http-client/interfaces/http-client.interfaces';
import { HttpData, HttpMethod } from '@/lib/http-client/types/http-client.types';
import { getRefreshPathByRequestPath } from '@/features/module-auth/lib/auth-refresh';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

export abstract class AbstractHttpClient {
    protected readonly baseUrl: string;
    protected readonly defaultHeaders: HeadersInit;
    protected readonly cache?: RequestCache;
    protected readonly next?: HttpClientOptions['next'];

    constructor(options: {
        baseUrl: string;
        cache: RequestCache;
        headers: HeadersInit;
        next: { revalidate: number };
    }) {
        this.baseUrl = options.baseUrl;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            Accept: 'application/json',
            ...options.headers,
        };
        this.cache = options.cache;
        this.next = options.next;
    }

    protected abstract request<TResponse, TBody = undefined>(
        path: string,
        method: HttpMethod,
        body?: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>>;

    get<TResponse>(path: string, config?: RequestConfig): Promise<HttpResponse<TResponse>> {
        return this.request<TResponse>(path, 'GET', undefined, config);
    }

    post<TResponse, TBody>(
        path: string,
        body: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>> {
        return this.request<TResponse, TBody>(path, 'POST', body, config);
    }

    put<TResponse, TBody>(
        path: string,
        body: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>> {
        return this.request<TResponse, TBody>(path, 'PUT', body, config);
    }

    patch<TResponse, TBody>(
        path: string,
        body: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>> {
        return this.request<TResponse, TBody>(path, 'PATCH', body, config);
    }

    delete<TResponse>(path: string, config?: RequestConfig): Promise<HttpResponse<TResponse>> {
        return this.request<TResponse>(path, 'DELETE', undefined, config);
    }

    protected async fetchData<TResponse>(
        options: RequestInit,
        url: URL,
        path: string,
        retry = true,
    ): Promise<HttpResponse<TResponse>> {
        let res = await fetch(url.toString(), options);

        if (res.status === 401 && retry) {
            const retryOptions = await this.refreshToken(path, options);

            if (retryOptions) {
                res = await fetch(url.toString(), retryOptions);
            }
        }

        const responseBody = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw {
                success: false,
                statusCode: res.status,
                error: responseBody?.error ?? res.statusText,
                message: responseBody?.message,
                timestamp: new Date().toISOString(),
                path,
            } satisfies HttpError;
        }

        return {
            ...(responseBody || {}),
            headers: res.headers,
        } as unknown as HttpResponse<TResponse>;
    }

    private async refreshToken(path: string, options: RequestInit): Promise<RequestInit | null> {
        if (path.includes('/refresh')) {
            return null;
        }
        try {
            const response = await fetch(
                this.getRefreshUrl(path),
                await this.getRefreshOptions(path, options),
            );

            if (!response.ok) {
                if (response.status === 401 && typeof document !== 'undefined') {
                    document.cookie = `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                    document.cookie = `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                    document.cookie = `${COOKIE_NAMES.ADMIN_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                    document.cookie = `${COOKIE_NAMES.ADMIN_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;

                    const currentPath = window.location.pathname;
                    const loginPath = currentPath.startsWith('/admin') ? '/admin/login' : '/login';
                    window.location.href = loginPath;
                }
                return null;
            }

            return await this.getRetryOptionsAfterRefresh(path, options, response);
        } catch {
            return null;
        }
    }

    protected getRefreshOptions(
        path: string,
        options?: RequestInit,
    ): RequestInit | Promise<RequestInit> {
        void path;
        void options;

        return {
            method: 'POST',
            credentials: 'include',
        };
    }

    protected getRetryOptionsAfterRefresh(
        path: string,
        options: RequestInit,
        response: Response,
    ): RequestInit | Promise<RequestInit> {
        void path;
        void response;

        return options;
    }

    protected getRefreshUrl(path: string): URL {
        return new URL(getRefreshPathByRequestPath(path), this.baseUrl);
    }
}
