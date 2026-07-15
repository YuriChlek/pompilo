import { applyDecorators, SetMetadata, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '@/module-auth/guards/jwt-auth.guard';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { RolesGuard } from '@/module-auth/guards/roles.guard';
import { ROLES_KEY } from '@/module-auth/constants/auth.constants';

export { ROLES_KEY } from '@/module-auth/constants/auth.constants';

export const Authorisation = (...roles: UserRoles[]) => {
    return applyDecorators(SetMetadata(ROLES_KEY, roles), UseGuards(JwtAuthGuard, RolesGuard));
};
