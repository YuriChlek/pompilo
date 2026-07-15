import {
    redactMailRecipient,
    redactMailRecipients,
} from '@/module-mail/utils/mail-log-redaction.util';

describe('mail log redaction', () => {
    it('masks an email address while retaining minimal operational context', () => {
        expect(redactMailRecipient('john.doe@example.com')).toBe('j***@e***.com');
    });

    it('masks every recipient in a recipient list', () => {
        const result = redactMailRecipients(['alice@example.com', 'bob@company.org']);

        expect(result).toBe('a***@e***.com, b***@c***.org');
        expect(result).not.toContain('alice@example.com');
        expect(result).not.toContain('bob@company.org');
    });

    it('does not echo malformed recipient input', () => {
        expect(redactMailRecipient('not-an-email')).toBe('[redacted-recipient]');
    });
});
