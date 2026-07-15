import { ConfigService } from '@nestjs/config';
import { EncryptService } from '@/module-encrypt/services/encrypt.service';

describe('EncryptService', () => {
    let service: EncryptService;
    let configService: {
        getOrThrow: jest.MockedFunction<ConfigService['getOrThrow']>;
    };

    beforeEach(() => {
        configService = {
            getOrThrow: jest.fn().mockReturnValue('unit-test-secret'),
        };

        service = new EncryptService(configService as unknown as ConfigService);
    });

    it('encrypts sensitive values and decrypts them back to the original text', () => {
        const plaintext = 'SensitivePayload';
        const ciphertext = service.encrypt(plaintext);

        expect(ciphertext).not.toEqual(plaintext);
        expect(service.decrypt(ciphertext)).toEqual(plaintext);
    });

    it('uses random initialization vectors so ciphertext changes each time', () => {
        const first = service.encrypt('repeatable');
        const second = service.encrypt('repeatable');

        expect(first).not.toEqual(second);
    });
});
