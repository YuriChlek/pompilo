import {
    CanActivate,
    ExecutionContext,
    Injectable,
    UnauthorizedException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { Request } from 'express';
import { timingSafeEqual } from 'crypto';

@Injectable()
export class ServiceTokenGuard implements CanActivate {
    constructor(private readonly configService: ConfigService) {}

    canActivate(context: ExecutionContext): boolean {
        const secret = this.configService.get<string>('SERVICE_TO_SERVICE_SECRET');
        if (!secret) {
            throw new UnauthorizedException('service_to_service_secret_not_configured');
        }

        const request = context.switchToHttp().getRequest<Request>();
        const headerValue = request.header('authorization') || request.header('x-service-token');
        const token = this.extractToken(headerValue);

        if (!token || !this.constantTimeEquals(token, secret)) {
            throw new UnauthorizedException('invalid_service_token');
        }

        return true;
    }

    private extractToken(value: string | undefined): string | null {
        if (!value) {
            return null;
        }

        const [scheme, token] = value.split(' ');
        if (scheme?.toLowerCase() === 'bearer' && token) {
            return token;
        }

        return value;
    }

    private constantTimeEquals(received: string, expected: string): boolean {
        const receivedBuffer = Buffer.from(received);
        const expectedBuffer = Buffer.from(expected);

        if (receivedBuffer.length !== expectedBuffer.length) {
            return false;
        }

        return timingSafeEqual(receivedBuffer, expectedBuffer);
    }
}
