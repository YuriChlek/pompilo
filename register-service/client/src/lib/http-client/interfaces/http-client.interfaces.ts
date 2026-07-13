import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';

export interface HttpClientOptions {
    baseUrl: string;
    headers?: HeadersInit;
    cache?: RequestCache;
    next?: {
        revalidate?: number;
        tags?: string[];
    };
}

export interface RequestConfig {
    params?: Record<string, string | number | boolean | undefined>;
    headers?: HeadersInit;
    cache?: RequestCache;
    next?: HttpClientOptions['next'];
    authScope?: AuthScope;
    timeout?: number;
    signal?: AbortSignal;
    retry?: boolean;
}

export interface HttpResponse<T> {
    data?: T;
    success: boolean;
    statusCode: number;
    message?: string;
    error?: string | string[] | null;
    //headers: Headers;
    timestamp: string;
}

export interface HttpError {
    success: false;
    statusCode?: number;
    message?: string;
    error: string | string[] | null;
    timestamp: string;
    path: string;
}
