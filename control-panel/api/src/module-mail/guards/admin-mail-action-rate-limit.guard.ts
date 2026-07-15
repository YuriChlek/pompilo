import {
    CanActivate,
    ExecutionContext,
    HttpException,
    HttpStatus,
    Injectable,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import type { Request, Response } from 'express';
import { createHash } from 'crypto';
import { RedisService } from '@/common/redis/redis.service';
import { ConfigService } from '@nestjs/config';
import {
    ADMIN_MAIL_ACTION_RATE_LIMIT_CONFIG,
    ADMIN_MAIL_ACTION_RATE_LIMIT_MESSAGE,
    ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY,
    AdminMailActionRateLimitOptions,
} from '@/module-mail/constants/admin-mail-action-rate-limit.constants';
import type { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { getCanonicalClientIp } from '@/common/utils/request-metadata.util';

const INCREMENT_WITH_TTL_SCRIPT = `
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return current
`;

@Injectable()
export class AdminMailActionRateLimitGuard implements CanActivate {
    private readonly redis: ReturnType<RedisService['getClient']>;

    constructor(
        private readonly reflector: Reflector,
        redisService: RedisService,
        private readonly configService: ConfigService,
    ) {
        this.redis = redisService.getClient();
    }

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const nodeEnv = this.configService.get<string>('NODE_ENV', 'development');
        if (nodeEnv !== 'production') {
            return true;
        }

        const options = this.reflector.getAllAndOverride<AdminMailActionRateLimitOptions>(
            ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY,
            [context.getHandler(), context.getClass()],
        );

        if (!options) {
            return true;
        }

        const request = context.switchToHttp().getRequest<Request>();
        const adminUserId = this.extractAdminUserId(request);
        const key = this.buildKey(options.action, adminUserId);
        const config = ADMIN_MAIL_ACTION_RATE_LIMIT_CONFIG[options.action];
        const maxRequests = config.maxRequests;
        const current = Number(
            await this.redis.eval(INCREMENT_WITH_TTL_SCRIPT, 1, key, config.windowSeconds * 1000),
        );

        if (current <= maxRequests) {
            return true;
        }

        const ttlMs = await this.redis.pttl(key);
        const retryAfterSeconds = ttlMs > 0 ? Math.ceil(ttlMs / 1000) : config.windowSeconds;
        const response = context.switchToHttp().getResponse<Response>();
        response.setHeader('Retry-After', String(retryAfterSeconds));

        throw new HttpException(
            {
                statusCode: HttpStatus.TOO_MANY_REQUESTS,
                message: ADMIN_MAIL_ACTION_RATE_LIMIT_MESSAGE,
            },
            HttpStatus.TOO_MANY_REQUESTS,
        );
    }

    private extractAdminUserId(request: Request): string {
        const user = request.user as AccessTokenPayload | undefined;
        if (user?.userId) {
            return user.userId;
        }

        const ipAddress = getCanonicalClientIp(request);

        return `ip:${ipAddress}`;
    }

    private buildKey(
        action: AdminMailActionRateLimitOptions['action'],
        identifier: string,
    ): string {
        const identifierHash = createHash('sha256').update(identifier).digest('hex');
        return `rate-limit:admin-mail:${action}:${identifierHash}`;
    }
}
