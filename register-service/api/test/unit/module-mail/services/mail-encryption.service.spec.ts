import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';

describe('MailEncryptionService', () => {
    let service: MailEncryptionService;
    let configService: jest.Mocked<ConfigService>;

    beforeEach(async () => {
        configService = {
            get: jest.fn(),
        } as unknown as jest.Mocked<ConfigService>;

        const module: TestingModule = await Test.createTestingModule({
            providers: [MailEncryptionService, { provide: ConfigService, useValue: configService }],
        }).compile();

        service = module.get<MailEncryptionService>(MailEncryptionService);
    });

    it('should encrypt and decrypt a secret successfully', () => {
        configService.get.mockReturnValue('test-encryption-key');
        const secret = 'super-secret-password';

        const encrypted = service.encryptMailSecret(secret);
        expect(encrypted).toContain(':');

        const decrypted = service.decryptMailSecret(encrypted);
        expect(decrypted).toBe(secret);
    });

    it('should throw error if encryption key is missing during encryption', () => {
        configService.get.mockReturnValue(null);
        expect(() => service.encryptMailSecret('secret')).toThrow(
            'MAIL_SETTINGS_ENCRYPTION_KEY is missing',
        );
    });

    it('should throw error if encryption key is missing during decryption', () => {
        configService.get.mockReturnValue(null);
        expect(() => service.decryptMailSecret('1:1:iv:tag:data')).toThrow(
            'MAIL_SETTINGS_ENCRYPTION_KEY is missing',
        );
    });

    it('should throw encryption_key_mismatch if decryption fails (e.g. wrong key)', () => {
        configService.get.mockReturnValue('key-1');
        const encrypted = service.encryptMailSecret('secret');

        configService.get.mockReturnValue('key-2');
        expect(() => service.decryptMailSecret(encrypted)).toThrow('encryption_key_mismatch');
    });

    it('should throw error if encrypted format is invalid', () => {
        configService.get.mockReturnValue('key');
        expect(() => service.decryptMailSecret('invalid-format')).toThrow(
            'Invalid encrypted secret format',
        );
    });

    it('should throw error if algo version is unsupported', () => {
        configService.get.mockReturnValue('key');
        expect(() => service.decryptMailSecret('999:1:iv:tag:data')).toThrow(
            'Unsupported encryption algorithm version',
        );
    });

    it('should mask secret presence correctly', () => {
        const settingsWithSecret = { smtpPasswordEncrypted: 'some-encrypted-data' };
        const masked = service.maskMailSecretPresence(settingsWithSecret) as {
            smtpPasswordEncrypted: string | null;
        };
        expect(masked.smtpPasswordEncrypted).toBe('********');

        const settingsWithoutSecret = { smtpPasswordEncrypted: null };
        const maskedNone = service.maskMailSecretPresence(settingsWithoutSecret) as {
            smtpPasswordEncrypted: string | null;
        };
        expect(maskedNone.smtpPasswordEncrypted).toBeNull();
    });
});
