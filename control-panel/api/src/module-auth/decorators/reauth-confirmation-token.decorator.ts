import { createParamDecorator, ExecutionContext } from '@nestjs/common';
import { Request } from 'express';

export const ReauthConfirmationToken = createParamDecorator(
    (_data: unknown, ctx: ExecutionContext): string | undefined => {
        const request = ctx.switchToHttp().getRequest<Request>();
        const token = request.headers['x-reauth-confirmation'];
        if (Array.isArray(token)) {
            return token[0];
        }
        return token;
    },
);
