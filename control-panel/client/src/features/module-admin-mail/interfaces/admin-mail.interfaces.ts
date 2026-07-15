export interface MailSettings {
    smtpHost: string;
    smtpPort: number;
    smtpSecure: boolean;
    smtpUser: string | null;
    smtpPasswordEncrypted: string | null;
    fromAddress: string;
    fromName: string;
    replyTo: string | null;
    clientPublicUrl: string | null;
    enabled: boolean;
}

export interface UpdateMailSettingsDto {
    smtpHost?: string;
    smtpPort?: number;
    smtpSecure?: boolean;
    smtpUser?: string;
    smtpPassword?: string;
    fromAddress?: string;
    fromName?: string;
    replyTo?: string;
    clientPublicUrl?: string;
    enabled?: boolean;
}

export interface SendTestEmailDto {
    to: string;
    templateId: string;
}

export interface SendTestEmailResponse {
    outboxId: string;
}
