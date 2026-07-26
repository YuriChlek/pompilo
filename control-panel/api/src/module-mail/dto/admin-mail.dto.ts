import { Transform } from 'class-transformer';
import {
    IsBoolean,
    IsEmail,
    IsInt,
    IsNotEmpty,
    IsOptional,
    IsString,
    Max,
    Min,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class UpdateMailSettingsDto {
    @ApiProperty({ example: 'smtp.example.com' })
    @IsString()
    @IsOptional()
    smtpHost?: string;

    @ApiProperty({ example: 587 })
    @IsInt()
    @Min(1)
    @Max(65535)
    @IsOptional()
    smtpPort?: number;

    @ApiProperty({ example: false })
    @IsBoolean()
    @IsOptional()
    smtpSecure?: boolean;

    @ApiProperty({ example: 'user@example.com' })
    @IsString()
    @IsOptional()
    smtpUser?: string;

    @ApiProperty({ example: 'password123' })
    @IsString()
    @IsOptional()
    smtpPassword?: string;

    @ApiProperty({ example: 'no-reply@example.com' })
    @IsEmail()
    @IsOptional()
    fromAddress?: string;

    @ApiProperty({ example: 'Pampilo' })
    @IsString()
    @IsOptional()
    fromName?: string;

    @ApiProperty({ example: 'support@example.com' })
    @IsEmail()
    @IsOptional()
    replyTo?: string;

    @ApiProperty({ example: 'https://identity.example.com' })
    @IsString()
    @IsOptional()
    clientPublicUrl?: string;

    @ApiProperty({ example: true })
    @IsBoolean()
    @IsOptional()
    enabled?: boolean;
}

export class SendTestEmailDto {
    @ApiProperty({ example: 'admin@example.com' })
    @Transform(({ value }: { value: unknown }) =>
        typeof value === 'string' ? value.trim() : value,
    )
    @IsEmail()
    @IsNotEmpty()
    to!: string;

    @ApiProperty({ example: 'security-alert' })
    @IsString()
    @IsNotEmpty()
    templateId!: string;
}
