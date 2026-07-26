import { AuthGuard } from '@nestjs/passport';
import { ExecutionContext, Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ModuleRef, Reflector } from '@nestjs/core';
import { Observable } from 'rxjs';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { IS_PUBLIC_KEY } from '@/module-auth/decorators/public.decorator';

@Injectable()
export class JwtAuthGuard extends AuthGuard(['admin-jwt', 'customer-jwt']) {
    constructor(
        private readonly redisTokenService: RedisTokenService,
        private readonly reflector: Reflector,
        private readonly moduleRef?: ModuleRef,
    ) {
        super();
    }

    override async canActivate(context: ExecutionContext): Promise<boolean> {
        const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
            context.getHandler(),
            context.getClass(),
        ]);
        if (isPublic) {
            return true;
        }

        const canActivateResult = super.canActivate(context);

        let result = false;
        if (canActivateResult instanceof Observable) {
            const { lastValueFrom } = await import('rxjs');
            result = await lastValueFrom(canActivateResult);
        } else {
            result = await canActivateResult;
        }

        if (!result) {
            return false;
        }

        const request = context.switchToHttp().getRequest<{
            user?: { userId: string; sessionId?: string; jti?: string };
        }>();
        const user = request.user;
        if (user) {
            if (!user.sessionId) {
                return false;
            }

            const redisTokenService =
                this.redisTokenService || this.moduleRef?.get(RedisTokenService, { strict: false });
            if (!redisTokenService) {
                return false;
            }

            try {
                if (user.jti) {
                    const isRevoked = await redisTokenService.isTokenRevoked(user.jti);
                    if (isRevoked) {
                        return false;
                    }
                }

                const isSessionRevoked = await redisTokenService.isSessionRevoked(user.sessionId);
                if (isSessionRevoked) {
                    return false;
                }
            } catch {
                throw new ServiceUnavailableException('Service Unavailable');
            }
        }

        return true;
    }
}
