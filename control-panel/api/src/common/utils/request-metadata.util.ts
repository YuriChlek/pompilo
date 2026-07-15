import type { Request } from 'express';
import { isIP } from 'net';

export interface RequestMetadata {
    ipAddress: string;
    userAgent: string;
}

export function getCanonicalClientIp(request: Request, fallback = 'unknown'): string {
    return (
        normalizeIpAddress(request.ip) ??
        normalizeIpAddress(request.socket.remoteAddress) ??
        fallback
    );
}

export function getRequestMetadata(request: Request): RequestMetadata {
    return {
        ipAddress: getCanonicalClientIp(request, ''),
        userAgent: getUserAgent(request),
    };
}

function getUserAgent(request: Request): string {
    const userAgent: unknown = request.headers['user-agent'];

    if (Array.isArray(userAgent)) {
        const firstUserAgent: unknown = userAgent[0];
        return typeof firstUserAgent === 'string' ? firstUserAgent : '';
    }

    return typeof userAgent === 'string' ? userAgent : '';
}

function normalizeIpAddress(value: string | undefined): string | undefined {
    if (!value) {
        return undefined;
    }

    const trimmed = value.trim();
    if (!trimmed || trimmed.includes(',')) {
        return undefined;
    }

    const withoutIpv6Brackets =
        trimmed.startsWith('[') && trimmed.endsWith(']') ? trimmed.slice(1, -1) : trimmed;
    const withoutIpv4MappedPrefix = withoutIpv6Brackets.startsWith('::ffff:')
        ? withoutIpv6Brackets.slice('::ffff:'.length)
        : withoutIpv6Brackets;

    return isIP(withoutIpv4MappedPrefix) ? withoutIpv4MappedPrefix : undefined;
}

export function parseUserAgent(userAgent: string): { os: string; browser: string } {
    let os = 'Unknown OS';
    let browser = 'Unknown Browser';

    const ua = userAgent.toLowerCase();

    // OS detection
    if (ua.includes('windows')) os = 'Windows';
    else if (ua.includes('macintosh') || ua.includes('mac os')) os = 'macOS';
    else if (ua.includes('android')) os = 'Android';
    else if (ua.includes('iphone') || ua.includes('ipad')) os = 'iOS';
    else if (ua.includes('linux')) os = 'Linux';

    // Browser detection
    if (ua.includes('firefox')) browser = 'Firefox';
    else if (ua.includes('chrome')) browser = 'Chrome';
    else if (ua.includes('safari')) browser = 'Safari';
    else if (ua.includes('edge')) browser = 'Edge';
    else if (ua.includes('opera') || ua.includes('opr')) browser = 'Opera';

    return { os, browser };
}
