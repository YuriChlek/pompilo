import { Text, Section, Link } from '@react-email/components';
import { MailLayoutTemplate } from './mail-layout.template';

interface PasswordResetTemplateProps {
    userName: string;
    resetLink: string;
    expiresInMinutes: number;
}

export const PasswordResetTemplate = ({
    userName,
    resetLink,
    expiresInMinutes,
}: PasswordResetTemplateProps) => (
    <MailLayoutTemplate previewText="Password reset request" heading="Reset your password">
        <Text style={text}>Hi {userName},</Text>
        <Text style={text}>
            Someone requested a password reset for your Pampilo account. If this was you, please
            click the link below to set a new password. This link will expire in {expiresInMinutes}{' '}
            minutes.
        </Text>
        <Section style={buttonContainer}>
            <Link style={button} href={resetLink}>
                Reset Password
            </Link>
        </Section>
        <Text style={text}>
            If the button above doesn't work, you can copy and paste the following link into your
            browser:
        </Text>
        <Text style={linkText}>
            <Link style={anchor} href={resetLink}>
                {resetLink}
            </Link>
        </Text>
        <Text style={text}>
            If you didn't request a password reset, you can safely ignore this email. Your password
            will remain unchanged.
        </Text>
    </MailLayoutTemplate>
);

const text = {
    color: '#333',
    fontSize: '16px',
    lineHeight: '24px',
    margin: '16px 0',
};

const buttonContainer = {
    margin: '24px 0',
    textAlign: 'center' as const,
};

const button = {
    backgroundColor: '#007ee6',
    borderRadius: '4px',
    color: '#fff',
    display: 'inline-block',
    fontSize: '16px',
    fontWeight: 'bold',
    padding: '12px 24px',
    textDecoration: 'none',
};

const linkText = {
    fontSize: '14px',
    color: '#888',
    wordBreak: 'break-all' as const,
};

const anchor = {
    color: '#007ee6',
};
