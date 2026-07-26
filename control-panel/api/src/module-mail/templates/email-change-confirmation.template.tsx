import { Text, Section } from '@react-email/components';
import { MailLayoutTemplate } from './mail-layout.template';

interface EmailChangeConfirmationTemplateProps {
    userName: string;
    newEmail: string;
    code: string;
    expiresInMinutes: number;
}

export const EmailChangeConfirmationTemplate = ({
    userName,
    newEmail,
    code,
    expiresInMinutes,
}: EmailChangeConfirmationTemplateProps) => (
    <MailLayoutTemplate previewText="Confirm your new email address" heading="Email address change">
        <Text style={text}>Hi {userName},</Text>
        <Text style={text}>
            We received a request to change the email address for your account to{' '}
            <b>{newEmail}</b>.
        </Text>
        <Text style={text}>
            Please use the following verification code to confirm this change. This code will expire
            in {expiresInMinutes} minutes.
        </Text>
        <Section style={codeContainer}>
            <Text style={codeText}>{code}</Text>
        </Section>
        <Text style={text}>
            If you didn't request this change, please contact our support team immediately or secure
            your account by changing your password.
        </Text>
    </MailLayoutTemplate>
);

const text = {
    color: '#333',
    fontSize: '16px',
    lineHeight: '24px',
    margin: '16px 0',
};

const codeContainer = {
    backgroundColor: '#f4f4f4',
    borderRadius: '4px',
    margin: '24px 0',
    padding: '16px',
    textAlign: 'center' as const,
};

const codeText = {
    fontSize: '32px',
    fontWeight: 'bold',
    letterSpacing: '4px',
    margin: '0',
    color: '#000',
};
