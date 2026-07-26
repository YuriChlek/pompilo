import type { DataPatch } from '../src/module-data-patch/types/data-patch.types';

const MAIL_SETTINGS = {
    smtpHost: 'localhost',
    smtpPort: 1025,
    smtpSecure: false,
    smtpUser: null,
    smtpPasswordEncrypted: null,
    fromAddress: 'no-reply@pampilo.local',
    fromName: 'Pampilo',
    replyTo: 'support@pampilo.local',
    clientPublicUrl: 'https://localhost',
    enabled: true,
} as const;

export const patch: DataPatch = {
    name: '202607050001-configure-mail-settings',
    description: 'Seed deterministic SMTP mail settings without environment bootstrap values.',
    async apply(context): Promise<void> {
        const { client, logger } = context;

        try {
            await client.query(
                `
                    insert into "mail_settings" (
                        "singleton_key",
                        "provider",
                        "smtp_host",
                        "smtp_port",
                        "smtp_secure",
                        "smtp_user",
                        "smtp_password_encrypted",
                        "from_address",
                        "from_name",
                        "reply_to",
                        "client_public_url",
                        "enabled",
                        "last_verified_at",
                        "last_verification_error",
                        "updated_at"
                    )
                    values (
                        true,
                        'smtp',
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10,
                        null,
                        null,
                        now()
                    )
                    on conflict ("singleton_key") do update
                    set
                        "provider" = excluded."provider",
                        "smtp_host" = excluded."smtp_host",
                        "smtp_port" = excluded."smtp_port",
                        "smtp_secure" = excluded."smtp_secure",
                        "smtp_user" = excluded."smtp_user",
                        "smtp_password_encrypted" = excluded."smtp_password_encrypted",
                        "from_address" = excluded."from_address",
                        "from_name" = excluded."from_name",
                        "reply_to" = excluded."reply_to",
                        "client_public_url" = excluded."client_public_url",
                        "enabled" = excluded."enabled",
                        "last_verified_at" = excluded."last_verified_at",
                        "last_verification_error" = excluded."last_verification_error",
                        "updated_at" = now()
                `,
                [
                    MAIL_SETTINGS.smtpHost,
                    MAIL_SETTINGS.smtpPort,
                    MAIL_SETTINGS.smtpSecure,
                    MAIL_SETTINGS.smtpUser,
                    MAIL_SETTINGS.smtpPasswordEncrypted,
                    MAIL_SETTINGS.fromAddress,
                    MAIL_SETTINGS.fromName,
                    MAIL_SETTINGS.replyTo,
                    MAIL_SETTINGS.clientPublicUrl,
                    MAIL_SETTINGS.enabled,
                ],
            );
        } catch (error) {
            logger.error('Data patch "202607050001-configure-mail-settings" failed.');
            throw error;
        }
    },
};
