import { ApiProperty } from '@nestjs/swagger';
import { IsEmail, IsNotEmpty, IsString, Length } from 'class-validator';

export class ChangePasswordDto {
    @ApiProperty({ description: 'Current password', example: 'oldPassword123' })
    @IsString()
    @IsNotEmpty()
    oldPassword!: string;

    @ApiProperty({ description: 'New password', example: 'newPassword123' })
    @IsString()
    @IsNotEmpty()
    @Length(6, 100, { message: 'Password must be between 6 and 100 characters' })
    newPassword!: string;
}

export class ResetPasswordRequestDto {
    @ApiProperty({
        description: 'Email address of the user requesting password reset',
        example: 'user@example.com',
    })
    @IsEmail()
    @IsNotEmpty()
    email!: string;
}

export class ResetPasswordConfirmDto {
    @ApiProperty({ description: 'Reset token received in email', example: 'abcdef123456...' })
    @IsString()
    @IsNotEmpty()
    token!: string;

    @ApiProperty({ description: 'New password to set', example: 'newPassword123' })
    @IsString()
    @IsNotEmpty()
    @Length(6, 100, { message: 'Password must be between 6 and 100 characters' })
    newPassword!: string;
}

export class ChangeEmailRequestDto {
    @ApiProperty({
        description: 'New email address to associate with the account',
        example: 'newemail@example.com',
    })
    @IsEmail()
    @IsNotEmpty()
    newEmail!: string;
}

export class ChangeEmailConfirmDto {
    @ApiProperty({ description: '6-digit verification code received in email', example: '123456' })
    @IsString()
    @IsNotEmpty()
    @Length(6, 6, { message: 'Verification code must be exactly 6 characters' })
    code!: string;
}
