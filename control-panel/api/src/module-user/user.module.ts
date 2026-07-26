import { Module } from '@nestjs/common';
import { UserService } from '@/module-user/services/user.service';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { UserUniquenessService } from '@/module-user/services/user-uniqueness.service';
import { UserRepository } from '@/module-user/repository/user.repository';
import { UserCleanupService } from '@/module-user/services/user-cleanup.service';

@Module({
    providers: [
        UserService,
        UserPasswordService,
        UserUniquenessService,
        UserRepository,
        UserCleanupService,
    ],
    exports: [UserService, UserRepository, UserPasswordService, UserUniquenessService],
})
export class UserModule {}
