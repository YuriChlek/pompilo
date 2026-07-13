import {
    HttpException,
    Injectable,
    InternalServerErrorException,
    UnauthorizedException,
} from '@nestjs/common';
import { AuthService } from '@/module-auth/services/auth.service';
import type { Request, Response } from 'express';
import { LoginAdminDto } from '@/module-admin-auth/dto/login-admin.dto';
import { UserRoles, COOKIE_NAMES } from '@/module-auth/enums/auth.enums';
import { User, CheckpointResponse } from '@/module-user/interfaces/user.interfaces';
import { VerifyCheckpointDto } from '@/module-auth/dto/verify-checkpoint.dto';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';

@Injectable()
export class AdminAuthService {
    private readonly adminRoles = [UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN];

    public constructor(private readonly authService: AuthService) {}

    async login(response: Response, request: Request, loginAdminDto: LoginAdminDto) {
        return this.authService.login(response, request, loginAdminDto, this.adminRoles);
    }

    async verifyLoginCheckpoint(
        response: Response,
        request: Request,
        verifyCheckpointDto: VerifyCheckpointDto,
    ) {
        return this.authService.verifyLoginCheckpoint(response, request, verifyCheckpointDto);
    }

    async resendLoginCheckpoint(
        response: Response,
        request: Request,
        resendCheckpointDto: ResendCheckpointDto,
    ): Promise<CheckpointResponse> {
        return this.authService.resendLoginCheckpoint(response, request, resendCheckpointDto);
    }

    async logout(request: Request, response: Response): Promise<void> {
        await this.authService.logout(request, response, UserRoles.PLATFORM_ADMIN);
    }

    async refreshAdminToken(response: Response, request: Request) {
        try {
            const success = await this.authService.refreshAccessToken(
                response,
                request,
                COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
            );
            if (!success) {
                throw new UnauthorizedException('Refresh token is invalid or revoked');
            }
            return true;
        } catch (error) {
            if (error instanceof HttpException) {
                throw error;
            }

            throw new InternalServerErrorException('Failed to refresh admin token');
        }
    }

    getMe(request: Request): User {
        return this.authService.getMe(request, UserRoles.PLATFORM_ADMIN);
    }
}
