import { CanActivate, ExecutionContext, ForbiddenException, Injectable } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ROLES_KEY } from '@/module-auth/constants/auth.constants';
import { UserRoles, deriveAuthRealmFromRole } from '@/module-auth/enums/auth.enums';
import { Request } from 'express';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';

@Injectable()
export class RolesGuard implements CanActivate {
    constructor(private readonly reflector: Reflector) {}

    canActivate(context: ExecutionContext): boolean {
        const requiredRoles: UserRoles[] = this.reflector.getAllAndOverride<UserRoles[]>(
            ROLES_KEY,
            [context.getHandler(), context.getClass()],
        );

        if (!requiredRoles || requiredRoles.length === 0) {
            return true;
        }

        const request: Request = context.switchToHttp().getRequest();
        const user = request?.user as AccessTokenPayload;

        if (!user || !user.role) {
            throw new ForbiddenException('User not authorized.');
        }

        // Verify compatibility of token realm with its roles
        const roles = Array.isArray(user.role) ? user.role : [user.role];
        for (const r of roles) {
            if (user.realm !== deriveAuthRealmFromRole(r as UserRoles)) {
                throw new ForbiddenException('Realm and role mismatch.');
            }
        }

        const userRole = user.role as UserRoles;
        const hasRole = requiredRoles.includes(userRole);

        if (!hasRole) {
            throw new ForbiddenException(`Access denied.`);
        }

        return true;
    }
}
