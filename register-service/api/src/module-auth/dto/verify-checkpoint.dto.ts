import { IsNotEmpty, IsString } from 'class-validator';

export class VerifyCheckpointDto {
    @IsString()
    @IsNotEmpty()
    checkpointToken: string;

    @IsString()
    @IsNotEmpty()
    code: string;
}
