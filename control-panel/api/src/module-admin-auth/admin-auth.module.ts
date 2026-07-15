import { Module } from '@nestjs/common';
import { AuthModule } from '@/module-auth/auth.module';
import { AdminAuthController } from '@/module-admin-auth/controllers/admin-auth.controller';
import { AdminAuthService } from '@/module-admin-auth/services/admin-auth.service';

@Module({
    imports: [AuthModule],
    controllers: [AdminAuthController],
    providers: [AdminAuthService],
})
export class AdminAuthModule {}
