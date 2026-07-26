import { Module } from '@nestjs/common';
import { AdminBotsController } from '@/module-admin-bots/controllers/admin-bots.controller';
import { AdminBotsService } from '@/module-admin-bots/services/admin-bots.service';

@Module({
    controllers: [AdminBotsController],
    providers: [AdminBotsService],
    exports: [AdminBotsService],
})
export class AdminBotsModule {}
