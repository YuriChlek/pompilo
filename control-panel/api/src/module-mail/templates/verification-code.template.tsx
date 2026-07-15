import { Text, Section } from '@react-email/components';
import { MailLayoutTemplate } from './mail-layout.template';

interface VerificationCodeTemplateProps {
    userName: string;
    code: string;
    expiresInMinutes: number;
}

export const VerificationCodeTemplate = ({
    userName,
    code,
    expiresInMinutes,
}: VerificationCodeTemplateProps) => (
    <MailLayoutTemplate previewText="Your verification code" heading="Confirm your email address">
        <Text style={text}>Hi {userName},</Text>
        <Text style={text}>
            Please use the following verification code to complete your request. This code will
            expire in {expiresInMinutes} minutes.
        </Text>
        <Section style={codeContainer}>
            <Text style={codeText}>{code}</Text>
        </Section>
        <Text style={text}>If you didn't request this code, you can safely ignore this email.</Text>
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
