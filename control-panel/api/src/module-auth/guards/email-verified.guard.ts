import { CanActivate, ExecutionContext, ForbiddenException, Injectable } from '@nestjs/common';
import { UserRepository } from '@/module-user/repository/user.repository';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import type { Request } from 'express';

@Injectable()
export class EmailVerifiedGuard implements CanActivate {
    constructor(private readonly userRepository: UserRepository) {}

    async canActivate(context: ExecutionContext): Promise<boolean> {
        const request = context
            .switchToHttp()
            .getRequest<Request & { user?: AccessTokenPayload }>();
        const user = request.user;

        if (!user || !user.userId) {
            throw new ForbiddenException('User not authenticated.');
        }

        // Admins bypass email verification
        if (user.role === 'platformAdmin' || user.role === 'superAdmin' || user.role === 'admin') {
            return true;
        }

        const dbUser = await this.userRepository.findById(user.userId);
        if (!dbUser) {
            throw new ForbiddenException('User not found.');
        }

        if (!dbUser.emailVerifiedAt) {
            throw new ForbiddenException(
                'Please verify your email address to access this resource.',
            );
        }

        return true;
    }
}
