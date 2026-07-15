import { Text, Section, Button } from '@react-email/components';
import { MailLayoutTemplate } from './mail-layout.template';
import * as React from 'react';

interface EmailVerificationTemplateProps {
    userName: string;
    verificationLink: string;
}

export const EmailVerificationTemplate = ({
    userName,
    verificationLink,
}: EmailVerificationTemplateProps) => (
    <MailLayoutTemplate previewText="Verify your email" heading="Verify your email address">
        <Text style={text}>Hi {userName},</Text>
        <Text style={text}>
            Thank you for registering! Please click the button below to verify your email address
            and activate your account.
        </Text>
        <Section style={btnContainer}>
            <Button href={verificationLink} style={button}>
                Verify Email
            </Button>
        </Section>
        <Text style={text}>
            If the button doesn't work, you can also copy and paste the following link into your
            browser:
        </Text>
        <Text style={linkText}>{verificationLink}</Text>
        <Text style={text}>If you didn't create an account, you can safely ignore this email.</Text>
    </MailLayoutTemplate>
);

const text = {
    color: '#333',
    fontSize: '16px',
    lineHeight: '24px',
    margin: '16px 0',
};

const btnContainer = {
    textAlign: 'center' as const,
    margin: '32px 0',
};

const button = {
    backgroundColor: '#0070f3',
    borderRadius: '5px',
    color: '#fff',
    fontSize: '16px',
    fontWeight: 'bold',
    textDecoration: 'none',
    textAlign: 'center' as const,
    display: 'inline-block',
    padding: '12px 24px',
};

const linkText = {
    fontSize: '14px',
    color: '#0070f3',
    wordBreak: 'break-all' as const,
};
