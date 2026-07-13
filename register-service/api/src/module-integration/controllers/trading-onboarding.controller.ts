import { Controller, Post, UseGuards } from '@nestjs/common';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { CurrentUser } from '@/module-auth/decorators/current-user.decorator';
import { EmailVerifiedGuard } from '@/module-auth/guards/email-verified.guard';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import type { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

@Controller('integration/trading')
export class TradingOnboardingController {
    constructor(private readonly identityIntegrationService: IdentityIntegrationService) {}

    @Post('onboarding-token')
    @UseGuards(EmailVerifiedGuard)
    @Authorisation(UserRoles.USER)
    createOnboardingToken(@CurrentUser() user: AccessTokenPayload) {
        return this.identityIntegrationService.createTradingOnboardingToken(user.userId);
    }
}
