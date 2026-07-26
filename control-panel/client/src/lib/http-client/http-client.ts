import { HttpResponse, RequestConfig } from '@/lib/http-client/interfaces/http-client.interfaces';
import { HttpData, HttpMethod } from '@/lib/http-client/types/http-client.types';
import { httpClientConfig } from '@/lib/http-client/config/http-client.config';
import { AbstractHttpClient } from '@/lib/http-client/abstract-http-client';

function resolveNextOptions(cache: RequestCache | undefined, next: RequestConfig['next']) {
    return cache === 'no-store' ? undefined : next;
}

class HttpClient extends AbstractHttpClient {
    protected async request<TResponse, TBody = undefined>(
        path: string,
        method: HttpMethod,
        body?: HttpData<TBody>,
        config?: RequestConfig,
    ): Promise<HttpResponse<TResponse>> {
        const requestBaseUrl =
            typeof window === 'undefined' ? this.baseUrl : window.location.origin;
        const url = new URL(`/api${path}`, requestBaseUrl);

        if (config?.params) {
            Object.entries(config.params).forEach(([key, value]) => {
                if (value !== undefined) {
                    url.searchParams.append(key, String(value));
                }
            });
        }

        const cache = config?.cache ?? this.cache;
        const options: RequestInit = {
            method,
            headers: {
                ...this.defaultHeaders,
                ...config?.headers,
            },
            body: body ? JSON.stringify(body) : undefined,
            credentials: 'include',
            cache,
            next: resolveNextOptions(cache, config?.next ?? this.next),
            signal: config?.signal,
        };

        return await this.fetchData(options, url, path, config?.retry);
    }

    protected getRefreshUrl(path: string): URL {
        const refreshUrl = super.getRefreshUrl(path);
        const requestBaseUrl =
            typeof window === 'undefined' ? this.baseUrl : window.location.origin;

        return new URL(`/api${refreshUrl.pathname}${refreshUrl.search}`, requestBaseUrl);
    }
}

export const apiClient = new HttpClient(httpClientConfig);
