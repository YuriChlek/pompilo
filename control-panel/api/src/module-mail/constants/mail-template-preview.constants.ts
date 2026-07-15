import { VerificationCodeTemplate } from '../templates/verification-code.template';
import { PasswordResetTemplate } from '../templates/password-reset.template';
import { SecurityAlertTemplate } from '../templates/security-alert.template';
import { EmailChangeConfirmationTemplate } from '../templates/email-change-confirmation.template';
import { EmailVerificationTemplate } from '../templates/email-verification.template';
import { MailTemplateRegistryItem } from '../interfaces/mail-template-preview.interfaces';

export const MAIL_TEMPLATE_PREVIEW_REGISTRY: Record<string, MailTemplateRegistryItem> = {
    'email-verification': {
        id: 'email-verification',
        name: 'Email Verification',
        description: 'Template for verifying the primary account email after registration',
        subject: 'Verify your email address',
        component: EmailVerificationTemplate,
        demoProps: {
            userName: 'John Doe',
            verificationLink: 'https://example.com/auth/verify-email?token=fake-token-123',
        },
    },
    'verification-code': {
        id: 'verification-code',
        name: 'Verification Code',
        description: 'Template for login and authentication checkpoint verification codes',
        subject: 'Confirm your email address',
        component: VerificationCodeTemplate,
        demoProps: {
            userName: 'John Doe',
            code: '123456',
            expiresInMinutes: 15,
        },
    },
    'password-reset': {
        id: 'password-reset',
        name: 'Password Reset',
        description: 'Template for password reset link',
        subject: 'Reset your password',
        component: PasswordResetTemplate,
        demoProps: {
            userName: 'John Doe',
            resetLink: 'https://example.com/auth/reset-password?token=fake-token-123',
            expiresInMinutes: 10,
        },
    },
    'security-alert': {
        id: 'security-alert',
        name: 'Security Alert',
        description: 'Template for security notifications and warnings',
        subject: 'Security Notification',
        component: SecurityAlertTemplate,
        demoProps: {
            userName: 'John Doe',
            alertType: 'New Device Login',
            details:
                'A login was detected from a new browser: Chrome 124.0 on Linux (IP: 192.0.2.1).',
            timestamp: '2026-07-01 18:30:00 UTC',
            resetPasswordLink: 'https://example.com/auth/forgot-password',
            reviewDevicesLink: 'https://example.com/account/security',
        },
    },
    'email-change-confirmation': {
        id: 'email-change-confirmation',
        name: 'Email Change Confirmation',
        description: 'Template for confirming email address change',
        subject: 'Email address change',
        component: EmailChangeConfirmationTemplate,
        demoProps: {
            userName: 'John Doe',
            newEmail: 'john.new@example.com',
            code: '654321',
            expiresInMinutes: 15,
        },
    },
};
