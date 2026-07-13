import { Injectable, UnauthorizedException } from '@nestjs/common';
import { createHash, randomBytes } from 'crypto';
import { ReauthConfirmationRepository } from '@/module-auth-token/repository/reauth-confirmation.repository';
import { UserRepository } from '@/module-user/repository/user.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class ReauthConfirmationService {
    constructor(
        private readonly reauthConfirmationRepository: ReauthConfirmationRepository,
        private readonly userRepository: UserRepository,
        private readonly securityEventService: SecurityEventService,
    ) {}

    private hash(value: string): string {
        return createHash('sha256').update(value).digest('hex');
    }

    async createReauthConfirmation(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        actionScope: string,
        password: string,
        metadata: { ipAddress: string; userAgent: string },
        now = new Date(),
    ): Promise<string> {
        // 1. Fetch user to verify password
        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new UnauthorizedException('User not found');
        }

        // 2. Compare password
        const isPasswordValid = await Argon2HashUtil.compare(password, user.password);

        if (!isPasswordValid) {
            // Write re_auth_failed event
            await this.securityEventService.recordReauthFailed({
                userId,
                sessionId,
                realm,
                ipAddress: metadata.ipAddress,
                userAgent: metadata.userAgent,
                metadata: {
                    actionScope,
                },
            });
            throw new UnauthorizedException('Invalid password');
        }

        // 3. Generate confirmation token
        const rawToken = randomBytes(32).toString('hex');
        const tokenHash = this.hash(rawToken);

        // Expires in 5 minutes (300 seconds)
        const expiresAt = new Date(now.getTime() + 5 * 60 * 1000);

        // 4. Save to DB
        await this.reauthConfirmationRepository.save({
            userId,
            realm,
            sessionId,
            actionScope,
            confirmationTokenHash: tokenHash,
            expiresAt,
            createdAt: now,
        });

        // 5. Write re_auth_passed event
        await this.securityEventService.recordReauthPassed({
            userId,
            sessionId,
            realm,
            ipAddress: metadata.ipAddress,
            userAgent: metadata.userAgent,
            metadata: {
                actionScope,
            },
        });

        return rawToken;
    }

    async validateReauthConfirmationToken(
        rawConfirmationToken: string,
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        actionScope: string,
        now = new Date(),
    ): Promise<boolean> {
        const hash = this.hash(rawConfirmationToken);
        const confirmation = await this.reauthConfirmationRepository.findByTokenHash(hash);

        if (!confirmation) {
            return false;
        }

        const isValid =
            confirmation.userId === userId &&
            confirmation.realm === realm &&
            confirmation.sessionId === sessionId &&
            confirmation.actionScope === actionScope &&
            confirmation.consumedAt === null &&
            confirmation.expiresAt.getTime() > now.getTime();

        return isValid;
    }

    async consumeReauthConfirmation(
        rawConfirmationToken: string,
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        actionScope: string,
        now = new Date(),
        transaction?: RepositoryTransaction,
    ): Promise<boolean> {
        const hash = this.hash(rawConfirmationToken);
        return await this.reauthConfirmationRepository.consumeByTokenHash(
            hash,
            userId,
            realm,
            sessionId,
            actionScope,
            now,
            transaction,
        );
    }

    async expireReauthConfirmations(retentionPeriodMs = 0, now = new Date()): Promise<number> {
        return this.reauthConfirmationRepository.deleteExpired(retentionPeriodMs, now);
    }
}
