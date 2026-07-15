import {
    BadRequestException,
    Injectable,
    NotFoundException,
    ServiceUnavailableException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { randomBytes, createHash, createHmac } from 'crypto';

import { UserRepository } from '@/module-user/repository/user.repository';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { AccountSecurityRepository } from '@/module-account/repository/account-security.repository';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { deriveAuthRealmFromRole, type AuthRealm } from '@/module-auth/enums/auth.enums';
import { SessionService } from '@/module-auth-token/services/session.service';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { parseUserAgent } from '@/common/utils/request-metadata.util';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

import {
    ChangePasswordDto,
    ResetPasswordConfirmDto,
} from '@/module-account/dto/account-settings.dto';

@Injectable()
export class AccountSettingsService {
    constructor(
        private readonly transactionRepository: TransactionRepository,
        private readonly accountSecurityRepository: AccountSecurityRepository,
        private readonly userRepository: UserRepository,
        private readonly userPasswordService: UserPasswordService,
        private readonly mailReadinessService: MailReadinessService,
        private readonly mailSettingsService: MailSettingsService,
        private readonly passwordResetChallengeRepository: PasswordResetChallengeRepository,
        private readonly emailChangeChallengeRepository: EmailChangeChallengeRepository,
        private readonly mailTemplateService: MailTemplateService,
        private readonly sessionService: SessionService,
        private readonly configService: ConfigService,
        private readonly reauthConfirmationService: ReauthConfirmationService,
        private readonly securityEventService: SecurityEventService,
        private readonly identityIntegrationService: IdentityIntegrationService,
    ) {}

    async getActiveSessions(userId: string, realm: AuthRealm, currentSessionId: string) {
        const activeSessions = await this.accountSecurityRepository.findActiveSessions(
            userId,
            realm,
        );

        return activeSessions.map(session => ({
            ...session,
            currentSession: session.id === currentSessionId,
        }));
    }

    async revokeSession(userId: string, realm: AuthRealm, sessionId: string): Promise<void> {
        await this.sessionService.revokeSpecificSession({
            userId,
            realm,
            sessionId,
        });
    }

    async revokeOtherSessions(
        userId: string,
        realm: AuthRealm,
        currentSessionId: string,
        reauthConfirmationToken?: string,
    ): Promise<void> {
        await this.sessionService.revokeOtherSessions({
            userId,
            realm,
            currentSessionId,
            reauthConfirmationToken,
        });
    }

    async revokeAllSessions(
        userId: string,
        realm: AuthRealm,
        currentSessionId: string,
    ): Promise<void> {
        await this.sessionService.revokeAllSessions({
            userId,
            realm,
            currentSessionId,
        });
    }

    async changePassword(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        dto: ChangePasswordDto,
        reauthConfirmationToken?: string,
        metadata?: { ipAddress?: string; userAgent?: string },
    ): Promise<void> {
        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new NotFoundException('User not found');
        }

        const isOldPasswordValid = await this.userPasswordService.comparePassword(
            dto.oldPassword,
            user.password,
        );
        if (!isOldPasswordValid) {
            throw new BadRequestException('Incorrect current password');
        }

        const newPasswordHash = await this.userPasswordService.hashPassword(dto.newPassword);

        await this.transactionRepository.run(async transaction => {
            await this.enforceReauth(
                userId,
                realm,
                sessionId,
                'password_change',
                reauthConfirmationToken,
                transaction,
            );

            // Policy: authenticated password change revokes all sessions across all realms.
            const revokedSessionCount = await this.sessionService.revokeUserSessions(
                { userId },
                transaction,
            );

            await this.userRepository.update(userId, { password: newPasswordHash }, transaction);

            await this.securityEventService.recordPasswordChanged(
                {
                    userId,
                    realm,
                    sessionId,
                    ipAddress: metadata?.ipAddress,
                    userAgent: metadata?.userAgent,
                    metadata: {
                        revokedSessionCount,
                    },
                },
                transaction,
            );

            await this.mailTemplateService.sendSecurityAlert(
                user.email,
                user.name || 'User',
                'Password Changed',
                `Your password was successfully changed. Request source IP: ${metadata?.ipAddress || 'unknown'}, User Agent: ${metadata?.userAgent || 'unknown'}.`,
                new Date().toISOString(),
                transaction,
            );
        });
    }

    async resetPasswordRequest(email: string): Promise<void> {
        // Phase 6/14: Check readiness BEFORE any user lookup or secret generation
        await this.mailReadinessService.assertMailReadyForCriticalFlow();

        const [user] = await this.userRepository.findByNameOrEmail('', email);
        if (!user) {
            // Enumeration-safe: return success even if email not found
            return;
        }

        const settings = await this.mailSettingsService.getCachedSettings();
        const clientUrl = settings?.clientPublicUrl;
        if (!clientUrl) {
            throw new ServiceUnavailableException('mail_settings_client_url_missing');
        }

        // Phase 19: Secure challenge generation (selector + verifier)
        const selector = randomBytes(16).toString('hex');
        const verifier = randomBytes(32).toString('hex');
        const verifierDigest = createHash('sha256').update(verifier).digest('hex');

        const resetLink = `${clientUrl}/auth/reset-password?token=${selector}.${verifier}`;

        await this.transactionRepository.run(async transaction => {
            // Phase 17: Enforce one-active-challenge policy
            await this.passwordResetChallengeRepository.invalidateUserChallenges(
                user.id,
                transaction,
            );

            // Phase 15/20: Create challenge and outbox atomically
            await this.passwordResetChallengeRepository.create(
                {
                    userId: user.id,
                    selector,
                    verifierDigest,
                    expiresAt: new Date(Date.now() + 15 * 60 * 1000), // 15 minutes
                },
                transaction,
            );

            await this.mailTemplateService.sendPasswordReset(
                user.email,
                user.name || 'User', // Fallback to 'User' if name is missing
                resetLink,
                15,
                transaction,
            );
        });
    }

    async resetPasswordConfirm(
        token: string,
        dto: ResetPasswordConfirmDto,
        metadata?: { ipAddress?: string; userAgent?: string },
    ): Promise<void> {
        const [selector, verifier] = token.split('.');
        if (!selector || !verifier) {
            throw new BadRequestException('Invalid reset token format');
        }

        const challenge =
            await this.passwordResetChallengeRepository.findActiveBySelector(selector);
        if (!challenge) {
            throw new BadRequestException('Invalid or expired reset token');
        }

        const verifierDigest = createHash('sha256').update(verifier).digest('hex');
        if (challenge.verifierDigest !== verifierDigest) {
            // Increment attempt count atomically on verifier mismatch
            await this.passwordResetChallengeRepository.incrementAttempts(challenge.id);
            throw new BadRequestException('Invalid or expired reset token');
        }

        const user = await this.userRepository.findById(challenge.userId);
        if (!user) {
            throw new NotFoundException('User not found');
        }

        const newPasswordHash = await this.userPasswordService.hashPassword(dto.newPassword);
        await this.transactionRepository.run(async transaction => {
            // Phase 20/22: Atomic consume
            const consumed = await this.passwordResetChallengeRepository.consume(
                challenge.id,
                transaction,
            );
            if (!consumed) {
                throw new BadRequestException('Challenge already used or expired');
            }

            await this.sessionService.revokeUserSessions(
                {
                    userId: user.id,
                    eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
                    fallbackRealm: deriveAuthRealmFromRole(user.role),
                },
                transaction,
            );

            await this.accountSecurityRepository.updatePassword(
                user.id,
                newPasswordHash,
                transaction,
            );

            await this.mailTemplateService.sendSecurityAlert(
                user.email,
                user.name || 'User',
                'Password Reset Completed',
                `Your password has been successfully reset. Request source IP: ${metadata?.ipAddress || 'unknown'}, User Agent: ${metadata?.userAgent || 'unknown'}.`,
                new Date().toISOString(),
                transaction,
            );
        });
    }

    private getCodeDigest(code: string): string {
        let pepper = this.configService.get<string>('ENCRYPTION_KEY');
        if (!pepper) {
            const nodeEnv =
                this.configService.get<string>('NODE_ENV') || process.env.NODE_ENV || 'development';
            if (nodeEnv === 'test') {
                pepper = 'default-pepper-for-tests';
            } else {
                throw new ServiceUnavailableException('encryption_key_missing');
            }
        }
        return createHmac('sha256', pepper).update(code).digest('hex');
    }

    private async enforceReauth(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        actionScope: string,
        token: string | undefined,
        transaction: RepositoryTransaction,
    ): Promise<void> {
        const isEnforcementEnabled = this.configService.get<boolean>(
            'REAUTH_ENFORCEMENT_ENABLED',
            false,
        );
        if (!isEnforcementEnabled) {
            return;
        }

        if (!token) {
            throw new BadRequestException('Re-authentication confirmation token is required');
        }

        const consumed = await this.reauthConfirmationService.consumeReauthConfirmation(
            token,
            userId,
            realm,
            sessionId,
            actionScope,
            new Date(),
            transaction,
        );

        if (!consumed) {
            throw new BadRequestException(
                'Invalid or expired re-authentication confirmation token',
            );
        }
    }

    async changeEmailRequest(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        newEmail: string,
        reauthConfirmationToken?: string,
        metadata?: { ipAddress?: string; userAgent?: string },
    ): Promise<void> {
        // Phase 6/14: Check readiness BEFORE any user lookup or secret generation
        await this.mailReadinessService.assertMailReadyForCriticalFlow();

        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new NotFoundException('User not found');
        }

        const existingUser = await this.userRepository.findByEmail(newEmail, userId);
        if (existingUser) {
            throw new BadRequestException('Email is already in use');
        }

        const code = Math.floor(100000 + Math.random() * 900000).toString();
        const codeDigest = this.getCodeDigest(code);

        await this.transactionRepository.run(async transaction => {
            await this.enforceReauth(
                userId,
                realm,
                sessionId,
                'email_change',
                reauthConfirmationToken,
                transaction,
            );

            // Phase 17: Enforce one-active-challenge policy
            await this.emailChangeChallengeRepository.invalidateUserChallenges(userId, transaction);

            // Phase 16/22: Create challenge and outbox atomically
            // Source of truth for new email is the challenge row
            await this.emailChangeChallengeRepository.create(
                {
                    userId,
                    newEmail,
                    codeDigest,
                    expiresAt: new Date(Date.now() + 15 * 60 * 1000), // 15 minutes
                },
                transaction,
            );

            await this.securityEventService.recordEmailChangeRequested(
                {
                    userId,
                    realm,
                    sessionId,
                    ipAddress: metadata?.ipAddress,
                    userAgent: metadata?.userAgent,
                    metadata: {
                        newEmailHash: createHash('sha256')
                            .update(newEmail.trim().toLowerCase())
                            .digest('hex'),
                    },
                },
                transaction,
            );

            await this.mailTemplateService.sendEmailChangeConfirmation(
                newEmail,
                user.name || 'User',
                newEmail,
                code,
                15,
                transaction,
            );

            const uaParsed = parseUserAgent(metadata?.userAgent || '');
            const details = `A request was made to change your email address to ${newEmail}. Device: ${uaParsed.os}, Browser: ${uaParsed.browser}. IP: ${metadata?.ipAddress || 'unknown'}. If this wasn't you, please reset your password immediately.`;
            await this.mailTemplateService.sendSecurityAlert(
                user.email,
                user.name || 'User',
                'Email Change Requested',
                details,
                new Date().toISOString(),
                transaction,
            );
        });
    }

    async changeEmailConfirm(userId: string, code: string): Promise<void> {
        const challenge = await this.emailChangeChallengeRepository.findActiveByUserId(userId);
        if (!challenge) {
            throw new BadRequestException('No pending email change request found');
        }

        const codeDigest = this.getCodeDigest(code);
        if (challenge.codeDigest !== codeDigest) {
            // Increment attempt count atomically on code mismatch
            await this.emailChangeChallengeRepository.incrementAttempts(challenge.id);
            throw new BadRequestException('Invalid or expired verification code');
        }

        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new NotFoundException('User not found');
        }

        await this.transactionRepository.run(async transaction => {
            // Phase 20/22: Atomic consume
            const consumed = await this.emailChangeChallengeRepository.consume(
                challenge.id,
                transaction,
            );
            if (!consumed) {
                throw new BadRequestException('Challenge already used or expired');
            }

            await this.accountSecurityRepository.updateEmail(
                userId,
                challenge.newEmail,
                transaction,
            );

            // Email is part of the authenticated identity. Revoke every active session so no
            // token issued with the previous identity can remain usable after the change.
            await this.sessionService.revokeUserSessions({ userId }, transaction);
        });
    }

    async deactivateAccount(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        reauthConfirmationToken?: string,
    ): Promise<void> {
        await this.transactionRepository.run(async transaction => {
            await this.enforceReauth(
                userId,
                realm,
                sessionId,
                'account_deactivate',
                reauthConfirmationToken,
                transaction,
            );
            await this.accountSecurityRepository.deactivateUser(userId, transaction);
            await this.identityIntegrationService.recordUserDisabled(userId, transaction);
        });

        await this.sessionService.revokeUserSessions({
            userId,
        });
    }

    async scheduleAccountDeletion(
        userId: string,
        realm: AuthRealm,
        sessionId: string,
        reauthConfirmationToken?: string,
    ): Promise<void> {
        const deletionDate = new Date();
        deletionDate.setDate(deletionDate.getDate() + 30);
        // Policy: scheduling deletion does not revoke sessions; deactivation/reset flows do.
        await this.transactionRepository.run(async transaction => {
            await this.enforceReauth(
                userId,
                realm,
                sessionId,
                'account_delete',
                reauthConfirmationToken,
                transaction,
            );
            await this.userRepository.update(
                userId,
                { deletionScheduledAt: deletionDate, accountStatus: 'PENDING_DELETION' },
                transaction,
            );
            await this.identityIntegrationService.recordUserDeleted(userId, transaction);
        });
    }
}
