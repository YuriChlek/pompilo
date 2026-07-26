import { IsNotEmpty, IsString } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class VerifyEmailDto {
    @ApiProperty({
        description: 'The verification token sent to the user email',
        example: 'a1b2c3d4e5...',
    })
    @IsString()
    @IsNotEmpty({ message: 'Token is required.' })
    token: string;
}
