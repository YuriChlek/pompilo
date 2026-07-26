import { IsEmail, IsEnum, IsNotEmpty, IsString, Length, Matches } from 'class-validator';
import { AuthPasswordDto } from '@/module-auth/dto/auth-password.dto';
import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import { UserRoles } from '@/module-auth/enums/auth.enums';

export class BaseRegisterDto extends AuthPasswordDto {
    @ApiProperty({
        description: 'Full name of the user',
        example: 'John Doe',
        minLength: 2,
        maxLength: 50,
    })
    @IsString()
    @IsNotEmpty({ message: 'Name is required.' })
    @Length(2, 50, { message: 'The name must be between 2 and 50 characters.' })
    name: string;

    @ApiProperty({
        description: 'Email address for registration and communication',
        example: 'john.doe@example.com',
        maxLength: 255,
        pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
    })
    @IsString({ message: 'Email must be a string.' })
    @IsNotEmpty({ message: 'Email is required.' })
    @IsEmail({}, { message: 'Invalid email format.' })
    @Length(5, 255, { message: 'Email must be between 5 and 255 characters.' })
    @Matches(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/, {
        message: 'Please provide a valid email address.',
    })
    @Type(() => String)
    email: string;
}

export class RegisterUserDto extends BaseRegisterDto {
    @ApiProperty({
        description: 'Requested user role',
        example: UserRoles.USER,
        required: true,
    })
    @IsNotEmpty({ message: 'Role is required.' })
    @IsEnum(UserRoles, {
        message: `Role must be one of: ${Object.values(UserRoles).join(', ')}`,
    })
    role: UserRoles;
}

export class RegisterDto extends BaseRegisterDto {}
