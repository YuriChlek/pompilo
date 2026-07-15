import {
    Body,
    Container,
    Head,
    Heading,
    Hr,
    Html,
    Preview,
    Section,
    Text,
} from '@react-email/components';
import * as React from 'react';

interface MailLayoutProps {
    previewText?: string;
    heading?: string;
    children: React.ReactNode;
}

export const MailLayoutTemplate = ({ previewText, heading, children }: MailLayoutProps) => (
    <Html>
        <Head />
        {previewText && <Preview>{previewText}</Preview>}
        <Body style={main}>
            <Container style={container}>
                <Section style={logoContainer}>
                    <Heading style={logoText}>Pampilo</Heading>
                </Section>
                {heading && <Heading style={h1}>{heading}</Heading>}
                <Section style={content}>{children}</Section>
                <Hr style={hr} />
                <Section style={footer}>
                    <Text style={footerText}>
                        © {new Date().getFullYear()} Pampilo. All rights reserved.
                    </Text>
                    <Text style={footerText}>
                        This is an automated message, please do not reply to this email.
                    </Text>
                </Section>
            </Container>
        </Body>
    </Html>
);

const main = {
    backgroundColor: '#f6f9fc',
    fontFamily:
        '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Ubuntu,sans-serif',
};

const container = {
    backgroundColor: '#ffffff',
    margin: '0 auto',
    padding: '20px 0 48px',
    marginBottom: '64px',
};

const logoContainer = {
    padding: '20px 40px',
};

const logoText = {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#007ee6',
    margin: '0',
};

const h1 = {
    color: '#333',
    fontSize: '24px',
    fontWeight: 'bold',
    padding: '0 40px',
    margin: '30px 0',
};

const content = {
    padding: '0 40px',
};

const hr = {
    borderColor: '#e6ebf1',
    margin: '20px 0',
};

const footer = {
    padding: '0 40px',
};

const footerText = {
    color: '#8898aa',
    fontSize: '12px',
    lineHeight: '16px',
    margin: '4px 0',
};
