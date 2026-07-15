import { Body, Controller, Post, Req, Res } from '@nestjs/common';
import { AdminAuthService } from '@/module-admin-auth/services/admin-auth.service';
import type { Request, Response } from 'express';
import { User, CheckpointResponse } from '@/module-user/interfaces/user.interfaces';
import { LoginAdminDto } from '@/module-admin-auth/dto/login-admin.dto';
import { VerifyCheckpointDto } from '@/module-auth/dto/verify-checkpoint.dto';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';

@Controller()
export class AdminAuthController {
    constructor(private readonly adminAuthService: AdminAuthService) {}

    @Post('login')
    login(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() loginAdminDto: LoginAdminDto,
    ): Promise<User | CheckpointResponse> {
        return this.adminAuthService.login(response, request, loginAdminDto);
    }

    @Post('checkpoint/verify')
    verifyCheckpoint(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() verifyCheckpointDto: VerifyCheckpointDto,
    ): Promise<User> {
        return this.adminAuthService.verifyLoginCheckpoint(response, request, verifyCheckpointDto);
    }

    @Post('checkpoint/resend')
    resendCheckpoint(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() resendCheckpointDto: ResendCheckpointDto,
    ): Promise<CheckpointResponse> {
        return this.adminAuthService.resendLoginCheckpoint(response, request, resendCheckpointDto);
    }

    @Post('logout')
    logout(@Res({ passthrough: true }) response: Response, @Req() request: Request) {
        return this.adminAuthService.logout(request, response);
    }

    @Post('refresh')
    refreshAdminToken(@Res({ passthrough: true }) response: Response, @Req() request: Request) {
        return this.adminAuthService.refreshAdminToken(response, request);
    }

    @Post('me')
    @Authorisation(UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN)
    getMe(@Req() request: Request) {
        return this.adminAuthService.getMe(request);
    }
}
