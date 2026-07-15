import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Request, Response, CookieOptions } from 'express';
import { randomUUID } from 'crypto';
import ms, { type StringValue } from 'ms';
import { COOKIE_NAMES } from '@/module-auth/enums/auth.enums';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

@Injectable()
export class DeviceIdService {
    public constructor(private readonly configService: ConfigService) {}

    validateUuid(uuid: string): boolean {
        return UUID_REGEX.test(uuid);
    }

    readDeviceId(request: Request): string | null {
        const cookies = this.getRequestCookies(request);
        const deviceId = cookies[COOKIE_NAMES.DEVICE_ID];
        if (deviceId && this.validateUuid(deviceId)) {
            return deviceId;
        }
        return null;
    }

    getOrCreateDeviceId(request: Request): { deviceId: string; isNew: boolean } {
        const existing = this.readDeviceId(request);
        if (existing) {
            return { deviceId: existing, isNew: false };
        }
        return { deviceId: randomUUID(), isNew: true };
    }

    setDeviceIdCookie(response: Response, deviceId: string): void {
        const cookieTtlStr = this.configService.getOrThrow<StringValue>('DEVICE_ID_COOKIE_TTL');
        const maxAge = ms(cookieTtlStr);
        const isProduction = this.configService.get('NODE_ENV') === 'production';
        const cookieDomain = this.configService.get<string>('COOKIE_DOMAIN');

        const options: CookieOptions = {
            httpOnly: true,
            secure: isProduction,
            sameSite: 'lax',
            path: '/',
            maxAge,
        };

        if (cookieDomain) {
            options.domain = cookieDomain;
        }

        response.cookie(COOKIE_NAMES.DEVICE_ID, deviceId, options);
    }

    private getRequestCookies(request: Request): Record<string, string | undefined> {
        const rawCookies = request.cookies as unknown as Record<string, unknown> | undefined;
        const parsedCookies = this.parseCookies(request.headers.cookie);
        const cookies: Record<string, string | undefined> = {};

        if (parsedCookies) {
            Object.assign(cookies, parsedCookies);
        }
        if (rawCookies) {
            for (const [key, value] of Object.entries(rawCookies)) {
                if (typeof value === 'string') {
                    cookies[key] = value;
                }
            }
        }
        return cookies;
    }

    private parseCookies(cookieHeader?: string): Record<string, string> {
        if (!cookieHeader) return {};
        return cookieHeader.split(';').reduce<Record<string, string>>((cookies, item) => {
            const [rawKey, ...rawValueParts] = item.trim().split('=');
            if (!rawKey) {
                return cookies;
            }
            cookies[decodeURIComponent(rawKey)] = decodeURIComponent(rawValueParts.join('='));
            return cookies;
        }, {});
    }
}
