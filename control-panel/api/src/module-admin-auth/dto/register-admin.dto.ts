import { BaseRegisterDto } from '@/module-auth/dto/register-user.dto';
import { ApiProperty } from '@nestjs/swagger';
import { IsIn, IsNotEmpty } from 'class-validator';
import { UserRoles } from '@/module-auth/enums/auth.enums';

export class RegisterAdminDto extends BaseRegisterDto {
    @ApiProperty({
        description: 'Requested admin role',
        example: UserRoles.PLATFORM_ADMIN,
        required: true,
    })
    @IsNotEmpty({ message: 'Role is required.' })
    @IsIn([UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN], {
        message: `Role must be one of: ${UserRoles.PLATFORM_ADMIN}, ${UserRoles.SUPER_ADMIN}`,
    })
    role: UserRoles;
}
