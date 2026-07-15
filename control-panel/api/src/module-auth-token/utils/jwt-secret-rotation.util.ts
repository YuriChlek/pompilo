import { ConfigService } from '@nestjs/config';
import { verify } from 'jsonwebtoken';
import { JWT_ALGORITHM } from '@/module-auth/constants/auth.constants';

export function getJwtSecrets(configService: ConfigService): string[] {
    const currentSecret = configService.getOrThrow<string>('JWT_SECRET');
    const previousSecrets =
        typeof configService.get === 'function'
            ? configService.get<string[] | string>('JWT_PREVIOUS_SECRETS')
            : undefined;
    const parsedPreviousSecrets = Array.isArray(previousSecrets)
        ? previousSecrets
        : parsePreviousSecrets(previousSecrets);

    return [currentSecret, ...parsedPreviousSecrets].filter(
        (secret, index, secrets) => secrets.indexOf(secret) === index,
    );
}

export function selectJwtSecretForToken(token: string, secrets: string[]): string {
    for (const secret of secrets) {
        try {
            verify(token, secret, {
                algorithms: [JWT_ALGORITHM],
                ignoreExpiration: true,
            });
            return secret;
        } catch {
            continue;
        }
    }

    return secrets[0];
}

function parsePreviousSecrets(value: string | undefined): string[] {
    if (!value) {
        return [];
    }

    return value
        .split(',')
        .map(secret => secret.trim())
        .filter(Boolean);
}
