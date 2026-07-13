import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class EncryptService {
    private readonly key: Buffer;

    constructor(private readonly configService: ConfigService) {
        const rawKey = this.configService.getOrThrow<string>('ENCRYPTION_KEY');
        this.key = crypto.createHash('sha256').update(rawKey).digest();
    }

    encrypt(text: string): string {
        const iv = crypto.randomBytes(16);

        const cipher = crypto.createCipheriv('aes-256-gcm', this.key, iv);

        const encrypted = Buffer.concat([cipher.update(text, 'utf8'), cipher.final()]);

        const authTag = cipher.getAuthTag();

        return Buffer.concat([iv, authTag, encrypted]).toString('base64');
    }

    decrypt(data: string): string {
        const buffer = Buffer.from(data, 'base64');

        const iv = buffer.subarray(0, 16);
        const authTag = buffer.subarray(16, 32);
        const encrypted = buffer.subarray(32);

        const decipher = crypto.createDecipheriv('aes-256-gcm', this.key, iv);
        decipher.setAuthTag(authTag);

        const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);

        return decrypted.toString('utf8');
    }
}
