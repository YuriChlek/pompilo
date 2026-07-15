import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { UserRepository } from '@/module-user/repository/user.repository';

@Injectable()
export class UserCleanupService {
    private readonly logger = new Logger(UserCleanupService.name);

    constructor(private readonly userRepository: UserRepository) {}

    @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT)
    async handleCron() {
        this.logger.log('Starting scheduled cleanup of users pending deletion');
        try {
            const deletedCount = await this.userRepository.deleteExpiredUsers();
            this.logger.log(
                `Successfully deleted ${deletedCount} users whose grace period expired`,
            );
        } catch (error) {
            this.logger.error('Failed to cleanup expired users', error);
        }
    }
}
