import { IsNotEmpty, IsString, MinLength, Matches } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class AuthPasswordDto {
    @ApiProperty({
        description: 'User password.',
        example: 'SecurePass123',
        minLength: 6,
    })
    @IsString()
    @IsNotEmpty({ message: 'Password is required.' })
    @MinLength(6, { message: 'Password must be at least 6 characters.' })
    @Matches(/^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]/, {
        message: 'Password must contain at least one letter and one number.',
    })
    password: string;
}
