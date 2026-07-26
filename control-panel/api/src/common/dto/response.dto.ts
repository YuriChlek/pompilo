import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsBoolean, IsDateString, IsNumber, IsOptional, IsString } from 'class-validator';

export class ResponseData<T> {
    @ApiProperty({ description: 'Success status of the response' })
    @IsBoolean()
    success: boolean;

    @ApiProperty({ description: 'HTTP status code' })
    @IsNumber()
    statusCode: number;

    @ApiPropertyOptional({
        description: 'Response message',
        oneOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }],
    })
    @IsOptional()
    @IsString({ each: true })
    message?: string | string[];

    @ApiPropertyOptional({
        description: 'Response data',
    })
    @IsOptional()
    data?: T;

    @IsOptional()
    @IsString({ each: true })
    error?: string | string[] | null;

    @ApiProperty({ description: 'Timestamp of the response' })
    @IsDateString()
    timestamp: string;

    @ApiPropertyOptional({ description: 'Request path' })
    @IsOptional()
    @IsString()
    path?: string;
}
