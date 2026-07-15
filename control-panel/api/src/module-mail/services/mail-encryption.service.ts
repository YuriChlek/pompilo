import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as crypto from 'crypto';
import { getMailBootstrapConfig } from '@config/mail-bootstrap.config';

@Injectable()
export class MailEncryptionService {
    private readonly logger = new Logger(MailEncryptionService.name);
    private readonly algorithm = 'aes-256-gcm';
    private readonly algoVersion = '1';
    private readonly keyVersion = '1';

    constructor(private readonly configService: ConfigService) {}

    isKeyAvailable(): boolean {
        return !!this.getKey();
    }

    private getKey(): Buffer | null {
        const rawKey = getMailBootstrapConfig(this.configService).encryptionKey;
        if (!rawKey) {
            return null;
        }
        return crypto.createHash('sha256').update(rawKey).digest();
    }

    encryptMailSecret(rawSecret: string): string {
        const key = this.getKey();
        if (!key) {
            throw new Error('MAIL_SETTINGS_ENCRYPTION_KEY is missing. Cannot encrypt secret.');
        }

        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipheriv(this.algorithm, key, iv);

        const encrypted = Buffer.concat([cipher.update(rawSecret, 'utf8'), cipher.final()]);
        const authTag = cipher.getAuthTag();

        // Format: version:keyVersion:iv:authTag:encryptedData
        return [
            this.algoVersion,
            this.keyVersion,
            iv.toString('base64'),
            authTag.toString('base64'),
            encrypted.toString('base64'),
        ].join(':');
    }

    decryptMailSecret(encryptedSecret: string): string {
        const key = this.getKey();
        if (!key) {
            throw new Error('MAIL_SETTINGS_ENCRYPTION_KEY is missing. Cannot decrypt secret.');
        }

        const parts = encryptedSecret.split(':');
        if (parts.length !== 5) {
            throw new Error('Invalid encrypted secret format.');
        }

        const [algoVersion, , ivBase64, authTagBase64, encryptedBase64] = parts;

        if (algoVersion !== this.algoVersion) {
            throw new Error(`Unsupported encryption algorithm version: ${algoVersion}`);
        }

        const iv = Buffer.from(ivBase64, 'base64');
        const authTag = Buffer.from(authTagBase64, 'base64');
        const encrypted = Buffer.from(encryptedBase64, 'base64');

        try {
            const decipher = crypto.createDecipheriv(this.algorithm, key, iv);
            decipher.setAuthTag(authTag);

            const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);
            return decrypted.toString('utf8');
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Decryption failed: ${message}`);
            throw new Error('encryption_key_mismatch');
        }
    }

    maskMailSecretPresence(settings: { smtpPasswordEncrypted?: string | null }): any {
        return {
            ...settings,
            smtpPasswordEncrypted: settings.smtpPasswordEncrypted ? '********' : null,
        };
    }
}
