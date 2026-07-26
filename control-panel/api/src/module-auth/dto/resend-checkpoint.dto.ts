import { IsNotEmpty, IsString } from 'class-validator';

export class ResendCheckpointDto {
    @IsString()
    @IsNotEmpty()
    checkpointToken: string;
}
