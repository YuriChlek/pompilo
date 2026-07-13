import { Controller, Get, Post, Body, Res, Req } from '@nestjs/common';
import type { Request, Response } from 'express';
import { User, CheckpointResponse } from '@/module-user/interfaces/user.interfaces';
import { LoginUserDto } from '@/module-auth/dto/login-user.dto';
import { VerifyCheckpointDto } from '@/module-auth/dto/verify-checkpoint.dto';
import { ResendCheckpointDto } from '@/module-auth/dto/resend-checkpoint.dto';
import { VerifyEmailDto } from '@/module-auth/dto/verify-email.dto';
import { CustomerAuthService } from '@/module-customer-auth/services/customer-auth.service';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { RegisterDto } from '@/module-auth/dto/register-user.dto';
import { EmailFlowRateLimit } from '@/common/rate-limiting/decorators/email-flow-rate-limit.decorator';
import type { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';

@Controller()
export class CustomerAuthController {
    constructor(private readonly customerAuthService: CustomerAuthService) {}

    @Post(['auth/register', 'register'])
    @EmailFlowRateLimit({ flow: 'registration', recipientBodyField: 'email' })
    register(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() registerDto: RegisterDto,
    ): Promise<User> {
        return this.customerAuthService.register(response, request, registerDto);
    }

    @Post(['auth/login', 'login'])
    login(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() loginUserDto: LoginUserDto,
    ): Promise<User | CheckpointResponse> {
        return this.customerAuthService.login(response, request, loginUserDto);
    }

    @Post(['auth/checkpoint/verify', 'checkpoint/verify'])
    verifyCheckpoint(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() verifyCheckpointDto: VerifyCheckpointDto,
    ): Promise<User> {
        return this.customerAuthService.verifyLoginCheckpoint(
            response,
            request,
            verifyCheckpointDto,
        );
    }

    @Post(['auth/checkpoint/resend', 'checkpoint/resend'])
    resendCheckpoint(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
        @Body() resendCheckpointDto: ResendCheckpointDto,
    ): Promise<CheckpointResponse> {
        return this.customerAuthService.resendLoginCheckpoint(
            response,
            request,
            resendCheckpointDto,
        );
    }

    @Post(['auth/logout', 'logout'])
    logout(@Res({ passthrough: true }) response: Response, @Req() request: Request) {
        return this.customerAuthService.logout(request, response);
    }

    @Post(['auth/refresh', 'refresh'])
    refresh(
        @Res({ passthrough: true }) response: Response,
        @Req() request: Request,
    ): Promise<boolean> {
        return this.customerAuthService.refresh(response, request);
    }

    @Post('me')
    @Authorisation(UserRoles.USER)
    getMe(@Req() request: Request) {
        return this.customerAuthService.getMe(request);
    }

    @Get('auth/me')
    @Authorisation(UserRoles.USER)
    getMeV1(@Req() request: Request) {
        return this.customerAuthService.getMe(request);
    }

    @Post(['auth/email/verify', 'verify-email'])
    verifyEmail(@Body() verifyEmailDto: VerifyEmailDto): Promise<boolean> {
        return this.customerAuthService.verifyEmail(verifyEmailDto.token);
    }

    @Post(['auth/email/resend', 'resend-verification'])
    @Authorisation(UserRoles.USER)
    @EmailFlowRateLimit({ flow: 'resend_verification' })
    resendVerification(@Req() request: Request): Promise<void> {
        const user = request.user as AccessTokenPayload;
        return this.customerAuthService.resendVerification(user.userId);
    }
}
