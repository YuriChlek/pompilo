import { ConflictException, Injectable } from '@nestjs/common';
import { UserRepository } from '@/module-user/repository/user.repository';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class UserUniquenessService {
    constructor(private readonly userRepository: UserRepository) {}

    async ensureUnique(
        email: string,
        name: string,
        excludeId?: string,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const existingUsers = await this.userRepository.findExistingForUniqueness(
            email,
            name,
            excludeId,
            transaction,
        );

        if (existingUsers.length > 0) {
            throw new ConflictException('User with this email or user name already exists.');
        }
    }
}
