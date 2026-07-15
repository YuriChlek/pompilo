import { Text, Section, Link } from '@react-email/components';
import { MailLayoutTemplate } from './mail-layout.template';

interface SecurityAlertTemplateProps {
    userName: string;
    alertType: string;
    details: string;
    timestamp: string;
    resetPasswordLink?: string;
    reviewDevicesLink?: string;
}

export const SecurityAlertTemplate = ({
    userName,
    alertType,
    details,
    timestamp,
    resetPasswordLink = 'https://localhost/auth/forgot-password',
    reviewDevicesLink = 'https://localhost/account/security',
}: SecurityAlertTemplateProps) => (
    <MailLayoutTemplate previewText="Security Alert" heading="Security Notification">
        <Text style={text}>Hi {userName},</Text>
        <Text style={text}>
            We've detected unusual activity or a security event on your Pampilo account.
        </Text>
        <Section style={alertBox}>
            <Text style={alertHeading}>{alertType}</Text>
            <Text style={alertDetails}>{details}</Text>
            <Text style={alertTime}>Time: {timestamp}</Text>
        </Section>
        <Text style={text}>
            If you do not recognize this activity, we recommend that you take immediate recovery
            action using the options below:
        </Text>
        <Section style={buttonContainer}>
            <Link style={primaryButton} href={resetPasswordLink}>
                Reset Password
            </Link>
            <span style={divider}>or</span>
            <Link style={secondaryButton} href={reviewDevicesLink}>
                Review Active Devices
            </Link>
        </Section>
        <Text style={textFooter}>If this was you, you can safely ignore this email.</Text>
    </MailLayoutTemplate>
);

const text = {
    color: '#333',
    fontSize: '16px',
    lineHeight: '24px',
    margin: '16px 0',
};

const textFooter = {
    color: '#666',
    fontSize: '14px',
    lineHeight: '22px',
    margin: '16px 0',
};

const alertBox = {
    backgroundColor: '#fff5f5',
    border: '1px solid #feb2b2',
    borderRadius: '4px',
    margin: '24px 0',
    padding: '16px',
};

const alertHeading = {
    color: '#c53030',
    fontSize: '18px',
    fontWeight: 'bold',
    margin: '0 0 8px',
};

const alertDetails = {
    color: '#333',
    fontSize: '14px',
    margin: '0 0 8px',
};

const alertTime = {
    color: '#718096',
    fontSize: '12px',
    margin: '0',
};

const buttonContainer = {
    margin: '24px 0',
    textAlign: 'center' as const,
};

const primaryButton = {
    backgroundColor: '#e53e3e',
    borderRadius: '4px',
    color: '#fff',
    display: 'inline-block',
    fontSize: '15px',
    fontWeight: 'bold',
    padding: '11px 20px',
    textDecoration: 'none',
    margin: '0 10px',
};

const secondaryButton = {
    backgroundColor: '#3182ce',
    borderRadius: '4px',
    color: '#fff',
    display: 'inline-block',
    fontSize: '15px',
    fontWeight: 'bold',
    padding: '11px 20px',
    textDecoration: 'none',
    margin: '0 10px',
};

const divider = {
    color: '#718096',
    fontSize: '14px',
    margin: '0 10px',
};
