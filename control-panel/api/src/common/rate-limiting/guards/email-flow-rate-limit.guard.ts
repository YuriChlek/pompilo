import {
    CanActivate,
    ExecutionContext,
    HttpException,
    HttpStatus,
    Injectable,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import type { Request, Response } from 'express';
import {
    EMAIL_FLOW_RATE_LIMIT_MESSAGE,
    EMAIL_FLOW_RATE_LIMIT_METADATA_KEY,
} from '@/common/rate-limiting/constants/email-flow-rate-limit.constants';
import { EmailFlowRateLimitService } from '@/common/rate-limiting/services/email-flow-rate-limit.service';
import type { EmailFlowRateLimitOptions } from '@/common/rate-limiting/interfaces/email-flow-rate-limit.interfaces';
import { getCanonicalClientIp } from '@/common/utils/request-metadata.util';

@Injectable()
export class EmailFlowRateLimitGuard implements CanActivate {
    constructor(
        private readonly reflector: Reflector,
        private readonly rateLimitService: EmailFlowRateLimitService,
    ) {}

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const options = this.reflector.getAllAndOverride<EmailFlowRateLimitOptions>(
            EMAIL_FLOW_RATE_LIMIT_METADATA_KEY,
            [context.getHandler(), context.getClass()],
        );

        if (!options) {
            return true;
        }

        const request = context.switchToHttp().getRequest<Request>();
        const result = await this.rateLimitService.check({
            flow: options.flow,
            ipAddress: this.extractIpAddress(request),
            recipientEmail: this.extractRecipientEmail(request, options.recipientBodyField),
        });

        if (!result.limited) {
            return true;
        }

        const response = context.switchToHttp().getResponse<Response>();
        response.setHeader('Retry-After', String(result.retryAfterSeconds));

        throw new HttpException(
            {
                statusCode: HttpStatus.TOO_MANY_REQUESTS,
                message: EMAIL_FLOW_RATE_LIMIT_MESSAGE,
            },
            HttpStatus.TOO_MANY_REQUESTS,
        );
    }

    private extractIpAddress(request: Request): string {
        return getCanonicalClientIp(request);
    }

    private extractRecipientEmail(
        request: Request,
        bodyField?: EmailFlowRateLimitOptions['recipientBodyField'],
    ): string | undefined {
        const authenticatedUser = request.user as { email?: unknown } | undefined;
        if (typeof authenticatedUser?.email === 'string') {
            return authenticatedUser.email;
        }
        if (!bodyField) {
            return undefined;
        }
        const body = request.body as Record<string, unknown> | undefined;
        const value = body?.[bodyField];

        return typeof value === 'string' ? value : undefined;
    }
}
