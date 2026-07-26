export type MailAuditAction =
    | 'mail_settings_updated'
    | 'mail_settings_disabled'
    | 'mail_test_email_requested';

export interface MailAuditPayload {
    changedFields?: string[];
    hasSmtpPasswordChange?: boolean;
    enabled?: boolean;
    errorCode?: string;
    outboxId?: string;
    recipientMatchesAdmin?: boolean;
}

export interface CreateMailAuditEventInput {
    action: MailAuditAction;
    adminUserId: string;
    payload: MailAuditPayload;
}
