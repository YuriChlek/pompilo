import { BadRequestException, ConflictException, Injectable } from '@nestjs/common';
import { createHash, randomBytes } from 'crypto';
import { EmailVerificationRepository } from '@/module-auth-token/repository/email-verification.repository';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { UserRepository } from '@/module-user/repository/user.repository';
import { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { TransactionRepository } from '@/module-drizzle/repository/transaction.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { UserModel } from '@/module-user/types/user.types';

@Injectable()
export class EmailVerificationService {
    constructor(
        private readonly emailVerificationRepository: EmailVerificationRepository,
        private readonly mailTemplateService: MailTemplateService,
        private readonly userRepository: UserRepository,
        private readonly transactionRepository: TransactionRepository,
        private readonly securityEventService: SecurityEventService,
    ) {}

    private hashToken(token: string): string {
        return createHash('sha256').update(token).digest('hex');
    }

    async createVerificationToken(
        userId: string,
        transaction?: RepositoryTransaction,
    ): Promise<string> {
        const rawToken = randomBytes(32).toString('hex');
        const tokenHash = this.hashToken(rawToken);
        const expiresAt = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes TTL

        await this.emailVerificationRepository.save(
            {
                userId,
                tokenHash,
                expiresAt,
            },
            transaction,
        );

        return rawToken;
    }

    async sendVerificationEmail(
        userId: string,
        email: string,
        userName: string,
        rawToken: string,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const clientOrigin =
            process.env.CLIENT_PUBLIC_URL || process.env.CLIENT_ORIGIN || 'http://localhost:3001';
        const verificationLink = `${clientOrigin}/verify-email?token=${rawToken}`;

        await this.mailTemplateService.sendEmailVerificationOrThrow(
            email,
            userName,
            verificationLink,
            transaction,
        );
    }

    async verifyEmail(
        rawToken: string,
        transaction?: RepositoryTransaction,
        onVerified?: (user: UserModel, transaction: RepositoryTransaction) => Promise<void>,
    ): Promise<boolean> {
        const tokenHash = this.hashToken(rawToken);
        const execute = async (activeTransaction: RepositoryTransaction): Promise<boolean> => {
            const verification = await this.emailVerificationRepository.findByTokenHash(
                tokenHash,
                activeTransaction,
            );

            if (!verification) {
                throw new BadRequestException('Invalid or expired verification token.');
            }

            if (verification.consumedAt) {
                throw new ConflictException('Email has already been verified using this token.');
            }

            if (verification.expiresAt.getTime() < Date.now()) {
                throw new BadRequestException('Verification token has expired.');
            }

            const user = await this.userRepository.findById(verification.userId, activeTransaction);
            if (!user) {
                throw new BadRequestException('User not found.');
            }

            if (user.emailVerifiedAt) {
                throw new ConflictException('Email is already verified.');
            }

            const verifiedAt = new Date();
            const consumed = await this.emailVerificationRepository.consume(
                verification.id,
                verifiedAt,
                activeTransaction,
            );
            if (!consumed) {
                throw new ConflictException('Email verification token was already consumed.');
            }

            await this.userRepository.update(
                user.id,
                { emailVerifiedAt: verifiedAt },
                activeTransaction,
            );
            await this.securityEventService.recordEmailVerified(
                {
                    userId: user.id,
                    realm: 'customer',
                    metadata: {},
                },
                activeTransaction,
            );
            if (onVerified) {
                await onVerified(user, activeTransaction);
            }

            return true;
        };

        if (transaction) {
            return await execute(transaction);
        }

        return await this.transactionRepository.run(execute);
    }

    async resendVerification(
        userId: string,
        email: string,
        userName: string,
        transaction?: RepositoryTransaction,
    ): Promise<void> {
        const user = await this.userRepository.findById(userId, transaction);
        if (!user) {
            throw new BadRequestException('User not found.');
        }

        if (user.emailVerifiedAt) {
            throw new ConflictException('Email is already verified.');
        }

        const activeVerification = await this.emailVerificationRepository.findActiveByUserId(
            userId,
            new Date(),
            transaction,
        );

        if (activeVerification) {
            const timeSinceCreation = Date.now() - activeVerification.createdAt.getTime();
            if (timeSinceCreation < 60 * 1000) {
                // 1 minute rate limit
                throw new BadRequestException(
                    'Please wait 1 minute before requesting another email.',
                );
            }
        }

        const rawToken = await this.createVerificationToken(userId, transaction);
        await this.sendVerificationEmail(userId, email, userName, rawToken, transaction);
    }
}
