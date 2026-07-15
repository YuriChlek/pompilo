export const apiBaseUrl =
    process.env.API_BASE_URL ??
    `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost'}:${process.env.NEXT_PUBLIC_API_PORT ?? '3000'}`;
