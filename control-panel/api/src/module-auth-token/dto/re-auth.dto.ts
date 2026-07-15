import { ApiProperty } from '@nestjs/swagger';
import { IsIn, IsNotEmpty, IsString } from 'class-validator';

export const REAUTH_ACTION_SCOPES = [
    'password_change',
    'email_change',
    'account_deactivate',
    'account_delete',
    'revoke_other_sessions',
] as const;

export type ReauthActionScope = (typeof REAUTH_ACTION_SCOPES)[number];

export class ReauthDto {
    @ApiProperty({ description: 'Current password', example: 'SecurePassword123' })
    @IsString()
    @IsNotEmpty()
    password!: string;

    @ApiProperty({
        description: 'Scope of the sensitive action requiring reauthentication',
        enum: REAUTH_ACTION_SCOPES,
        example: 'password_change',
    })
    @IsString()
    @IsNotEmpty()
    @IsIn(REAUTH_ACTION_SCOPES, {
        message: `actionScope must be one of: ${REAUTH_ACTION_SCOPES.join(', ')}`,
    })
    actionScope!: ReauthActionScope;
}
