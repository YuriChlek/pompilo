import { createParamDecorator, ExecutionContext } from '@nestjs/common';
import { User } from '@/module-user/interfaces/user.interfaces';

export const CurrentUser = createParamDecorator((_data: unknown, ctx: ExecutionContext): User => {
    const request = ctx.switchToHttp().getRequest<Record<string, unknown>>();
    return request.user as User;
});
