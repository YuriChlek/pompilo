import { ArrayMaxSize, IsArray, IsInt, IsOptional, IsUUID, Max, Min } from 'class-validator';
import { Type } from 'class-transformer';

export class ListIdentityOutboxEventsDto {
    @IsOptional()
    @Type(() => Number)
    @IsInt()
    @Min(1)
    @Max(100)
    limit?: number;
}

export class AckIdentityOutboxEventsDto {
    @IsArray()
    @ArrayMaxSize(100)
    @IsUUID('4', { each: true })
    eventIds!: string[];
}
