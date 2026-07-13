const REDACTED_RECIPIENT = '[redacted-recipient]';

export function redactMailRecipient(recipient: string): string {
    const normalized = recipient.trim();
    const separatorIndex = normalized.lastIndexOf('@');
    if (separatorIndex <= 0 || separatorIndex === normalized.length - 1) {
        return REDACTED_RECIPIENT;
    }

    const localPart = normalized.slice(0, separatorIndex);
    const domain = normalized.slice(separatorIndex + 1);
    const domainParts = domain.split('.');
    const domainName = domainParts.shift() || '';
    const suffix = domainParts.length > 0 ? `.${domainParts.join('.')}` : '';

    if (!domainName) {
        return REDACTED_RECIPIENT;
    }

    return `${localPart.slice(0, 1)}***@${domainName.slice(0, 1)}***${suffix}`;
}

export function redactMailRecipients(recipients: string | string[]): string {
    const values = Array.isArray(recipients) ? recipients : [recipients];
    return values.map(redactMailRecipient).join(', ');
}
