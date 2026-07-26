import { sign } from 'jsonwebtoken';
import {
    getJwtSecrets,
    selectJwtSecretForToken,
} from '@/module-auth-token/utils/jwt-secret-rotation.util';
import type { ConfigService } from '@nestjs/config';

describe('jwt secret rotation utils', () => {
    const currentSecret = 'c'.repeat(32);
    const previousSecret = 'p'.repeat(32);

    it('returns current secret followed by unique previous secrets', () => {
        const configService = {
            getOrThrow: jest.fn().mockReturnValue(currentSecret),
            get: jest.fn().mockReturnValue(`${previousSecret},${previousSecret}`),
        } as unknown as ConfigService;

        expect(getJwtSecrets(configService)).toEqual([currentSecret, previousSecret]);
    });

    it('selects the previous secret for a token signed before rotation', () => {
        const token = sign({ sub: 'user-id' }, previousSecret, { algorithm: 'HS256' });

        expect(selectJwtSecretForToken(token, [currentSecret, previousSecret])).toBe(
            previousSecret,
        );
    });
});
