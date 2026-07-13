import {
    Inject,
    Injectable,
    Logger,
    ServiceUnavailableException,
    forwardRef,
} from '@nestjs/common';
import * as React from 'react';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import type { MailService } from '@/module-mail/interfaces/mail-service.interface';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { VerificationCodeTemplate } from '@/module-mail/templates/verification-code.template';
import { PasswordResetTemplate } from '@/module-mail/templates/password-reset.template';
import { SecurityAlertTemplate } from '@/module-mail/templates/security-alert.template';
import { EmailChangeConfirmationTemplate } from '@/module-mail/templates/email-change-confirmation.template';
import { EmailVerificationTemplate } from '@/module-mail/templates/email-verification.template';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { redactMailRecipient } from '@/module-mail/utils/mail-log-redaction.util';

@Injectable()
export class MailTemplateService {
    private readonly logger = new Logger(MailTemplateService.name);

    constructor(
        @Inject(MAIL_SERVICE)
        private readonly mailService: MailService,
        private readonly mailRenderService: MailRenderService,
        @Inject(forwardRef(() => MailSettingsService))
        private readonly mailSettingsService: MailSettingsService,
    ) {}

    async sendEmailVerification(
        to: string,
        userName: string,
        verificationLink: string,
        transaction?: RepositoryTransaction,
    ) {
        try {
            const component = React.createElement(EmailVerificationTemplate, {
                userName,
                verificationLink,
            });

            const html = await this.mailRenderService.renderHtml(component);
            const text = await this.mailRenderService.renderText(component);

            return await this.mailService.createDeliveryRequest(
                {
                    to,
                    subject: 'Verify your email address',
                    html,
                    text,
                },
                transaction,
            );
        } catch (error) {
            this.logger.error(
                `Failed to create delivery request for email verification [to=${redactMailRecipient(to)}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            return null;
        }
    }

    async sendEmailVerificationOrThrow(
        to: string,
        userName: string,
        verificationLink: string,
        transaction?: RepositoryTransaction,
    ) {
        const deliveryRequest = await this.sendEmailVerification(
            to,
            userName,
            verificationLink,
            transaction,
        );

        if (!deliveryRequest) {
            throw new ServiceUnavailableException('Verification email could not be queued.');
        }

        return deliveryRequest;
    }

    async sendVerificationCode(
        to: string,
        userName: string,
        code: string,
        expiresInMinutes = 15,
        transaction?: RepositoryTransaction,
    ) {
        try {
            const component = React.createElement(VerificationCodeTemplate, {
                userName,
                code,
                expiresInMinutes,
            });

            const html = await this.mailRenderService.renderHtml(component);
            const text = await this.mailRenderService.renderText(component);

            return await this.mailService.createDeliveryRequest(
                {
                    to,
                    subject: 'Your verification code',
                    html,
                    text,
                },
                transaction,
            );
        } catch (error) {
            this.logger.error(
                `Failed to create delivery request for verification code [to=${redactMailRecipient(to)}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            return null;
        }
    }

    async sendVerificationCodeOrThrow(
        to: string,
        userName: string,
        code: string,
        expiresInMinutes = 15,
        transaction?: RepositoryTransaction,
    ) {
        const deliveryRequest = await this.sendVerificationCode(
            to,
            userName,
            code,
            expiresInMinutes,
            transaction,
        );

        if (!deliveryRequest) {
            throw new ServiceUnavailableException('Verification email could not be queued.');
        }

        return deliveryRequest;
    }

    async sendPasswordReset(
        to: string,
        userName: string,
        resetLink: string,
        expiresInMinutes = 30,
        transaction?: RepositoryTransaction,
    ) {
        try {
            const component = React.createElement(PasswordResetTemplate, {
                userName,
                resetLink,
                expiresInMinutes,
            });

            const html = await this.mailRenderService.renderHtml(component);
            const text = await this.mailRenderService.renderText(component);

            return await this.mailService.createDeliveryRequest(
                {
                    to,
                    subject: 'Reset your password',
                    html,
                    text,
                },
                transaction,
            );
        } catch (error) {
            this.logger.error(
                `Failed to create delivery request for password reset [to=${redactMailRecipient(to)}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            return null;
        }
    }

    async sendSecurityAlert(
        to: string,
        userName: string,
        alertType: string,
        details: string,
        timestamp = new Date().toISOString(),
        transaction?: RepositoryTransaction,
        resetPasswordLink?: string,
        reviewDevicesLink?: string,
    ) {
        try {
            const settings = await this.mailSettingsService.getCachedSettings();
            const clientUrl = settings?.clientPublicUrl || 'http://localhost:3000';

            const finalResetLink = resetPasswordLink || `${clientUrl}/auth/forgot-password`;
            const finalReviewLink = reviewDevicesLink || `${clientUrl}/account/security`;

            const component = React.createElement(SecurityAlertTemplate, {
                userName,
                alertType,
                details,
                timestamp,
                resetPasswordLink: finalResetLink,
                reviewDevicesLink: finalReviewLink,
            });

            const html = await this.mailRenderService.renderHtml(component);
            const text = await this.mailRenderService.renderText(component);

            return await this.mailService.createDeliveryRequest(
                {
                    to,
                    subject: 'Security Alert',
                    html,
                    text,
                },
                transaction,
            );
        } catch (error) {
            this.logger.error(
                `Failed to create delivery request for security alert [type=${alertType}, to=${redactMailRecipient(to)}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            return null;
        }
    }

    async sendEmailChangeConfirmation(
        to: string,
        userName: string,
        newEmail: string,
        code: string,
        expiresInMinutes = 15,
        transaction?: RepositoryTransaction,
    ) {
        try {
            const component = React.createElement(EmailChangeConfirmationTemplate, {
                userName,
                newEmail,
                code,
                expiresInMinutes,
            });

            const html = await this.mailRenderService.renderHtml(component);
            const text = await this.mailRenderService.renderText(component);

            return await this.mailService.createDeliveryRequest(
                {
                    to,
                    subject: 'Confirm your new email address',
                    html,
                    text,
                },
                transaction,
            );
        } catch (error) {
            this.logger.error(
                `Failed to create delivery request for email change confirmation [to=${redactMailRecipient(to)}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            return null;
        }
    }
}
