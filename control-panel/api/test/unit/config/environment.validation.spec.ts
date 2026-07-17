import { validateEnvironment, validateLoggerEnvironment } from '@/config/environment.validation';

describe('validateEnvironment', () => {
    const validEnvironment = {
        COOKIE_DOMAIN: 'localhost',
        DB_HOST: 'localhost',
        DB_PORT: '5432',
        DB_USER: 'app',
        DB_PASSWORD: 'database-password',
        DB_NAME: 'identity_service',
        JWT_SECRET: 'j'.repeat(32),
        JWT_ACCESS_TOKEN_TTL: '15m',
        JWT_REFRESH_TOKEN_TTL: '7d',
        SESSION_MAX_TTL: '7d',
        DEVICE_ID_COOKIE_TTL: '365d',
        ENCRYPTION_KEY: 'e'.repeat(32),
    };

    function getValidationError(environment: Record<string, unknown>): Error {
        try {
            validateEnvironment(environment);
        } catch (error) {
            if (error instanceof Error) {
                return error;
            }
        }

        throw new Error('Expected environment validation to fail');
    }

    it('normalizes defaults and configured primitive values', () => {
        const environment = validateEnvironment({
            ...validEnvironment,
            NODE_ENV: 'test',
            PORT: '3100',
            REDIS_PORT: '6380',
            REDIS_DB: '2',
            MAIL_RETRY_BACKOFF_JITTER: '0.35',
            AUTH_CLOCK_SKEW_SECONDS: '45',
        });

        expect(environment).toEqual(
            expect.objectContaining({
                NODE_ENV: 'test',
                PORT: 3100,
                CLIENT_ORIGIN: 'http://localhost:3001',
                CLIENT_PUBLIC_URL: 'https://localhost',
                DB_PORT: 5432,
                DB_POOL_MAX: 30,
                REDIS_HOST: 'localhost',
                REDIS_PORT: 6380,
                REDIS_DB: 2,
                MAIL_QUEUE_LIMITER_MAX: 100,
                MAIL_QUEUE_LIMITER_DURATION: 1000,
                MAIL_WORKER_CONCURRENCY: 4,
                MAIL_RETRY_ATTEMPTS: 3,
                MAIL_RETRY_BACKOFF_DELAY: 2000,
                MAIL_RETRY_BACKOFF_TYPE: 'exponential',
                MAIL_RETRY_BACKOFF_JITTER: 0.35,
                SESSION_MAX_TTL: '7d',
                DEVICE_ID_COOKIE_TTL: '365d',
                AUTH_CLOCK_SKEW_SECONDS: 45,
                JWT_PREVIOUS_SECRETS: [],
                DEVICE_ID_FORWARDING_ENABLED: false,
                REFRESH_ROTATION_ENABLED: false,
                DEVICE_BINDING_REQUIRED_CUSTOMER: false,
                DEVICE_BINDING_REQUIRED_ADMIN: false,
                LOGIN_CHECKPOINT_ENABLED: false,
                REAUTH_ENFORCEMENT_ENABLED: false,
                CSRF_ORIGIN_CHECK_ENABLED: false,
                TRUST_PROXY: 'false',
                GEOIP_TIMEOUT_MS: 1000,
                LOG_LEVEL: 'debug',
                LOG_FORMAT: 'json',
                LOG_OUTPUT: 'console',
                LOG_FILE_DIR: 'logs',
                LOG_FILE_MAX_DAYS: 14,
                LOG_REDACTION_ENABLED: true,
                LOG_STRUCTURED_BOOTSTRAP_ENABLED: false,
                LOG_REQUEST_ID_HEADER: 'x-request-id',
                SERVICE_NAME: 'pampilo-api',
            }),
        );
    });

    it('normalizes logger defaults by environment', () => {
        const developmentLoggerEnvironment = validateLoggerEnvironment({
            NODE_ENV: 'development',
        });
        const productionLoggerEnvironment = validateLoggerEnvironment({
            NODE_ENV: 'production',
        });

        expect(developmentLoggerEnvironment).toEqual(
            expect.objectContaining({
                LOG_LEVEL: 'debug',
                LOG_FORMAT: 'json',
                LOG_OUTPUT: 'console',
                LOG_FILE_DIR: 'logs',
                LOG_FILE_MAX_DAYS: 14,
                LOG_REDACTION_ENABLED: true,
                LOG_STRUCTURED_BOOTSTRAP_ENABLED: false,
                LOG_REQUEST_ID_HEADER: 'x-request-id',
                SERVICE_NAME: 'pampilo-api',
            }),
        );
        expect(productionLoggerEnvironment).toEqual(
            expect.objectContaining({
                LOG_LEVEL: 'info',
                LOG_FORMAT: 'json',
                LOG_OUTPUT: 'console',
                LOG_FILE_DIR: 'logs',
                LOG_FILE_MAX_DAYS: 14,
                LOG_REDACTION_ENABLED: true,
                LOG_STRUCTURED_BOOTSTRAP_ENABLED: true,
                LOG_REQUEST_ID_HEADER: 'x-request-id',
                SERVICE_NAME: 'pampilo-api',
            }),
        );
    });

    it('validates configured logger values', () => {
        const environment = validateLoggerEnvironment({
            NODE_ENV: 'test',
            LOG_LEVEL: 'trace',
            LOG_FORMAT: 'json',
            LOG_OUTPUT: 'file',
            LOG_FILE_DIR: 'runtime-logs',
            LOG_FILE_MAX_DAYS: '30',
            LOG_REDACTION_ENABLED: 'false',
            LOG_STRUCTURED_BOOTSTRAP_ENABLED: 'true',
            LOG_REQUEST_ID_HEADER: 'x-correlation-id',
            SERVICE_NAME: 'pampilo-api-test',
        });

        expect(environment).toEqual({
            NODE_ENV: 'test',
            LOG_LEVEL: 'trace',
            LOG_FORMAT: 'json',
            LOG_OUTPUT: 'file',
            LOG_FILE_DIR: 'runtime-logs',
            LOG_FILE_MAX_DAYS: 30,
            LOG_REDACTION_ENABLED: false,
            LOG_STRUCTURED_BOOTSTRAP_ENABLED: true,
            LOG_REQUEST_ID_HEADER: 'x-correlation-id',
            SERVICE_NAME: 'pampilo-api-test',
        });
    });

    it('uses default AUTH_CLOCK_SKEW_SECONDS when not configured', () => {
        const environment = validateEnvironment({
            ...validEnvironment,
        });

        expect(environment.AUTH_CLOCK_SKEW_SECONDS).toBe(30);
    });

    it('validates optional bot platform base URL', () => {
        const environment = validateEnvironment({
            ...validEnvironment,
            BOT_PLATFORM_BASE_URL: 'http://bot_platform:8092',
        });

        expect(environment.BOT_PLATFORM_BASE_URL).toBe('http://bot_platform:8092');

        const error = getValidationError({
            ...validEnvironment,
            BOT_PLATFORM_BASE_URL: 'bot_platform:8092',
        });
        expect(error.message).toContain('BOT_PLATFORM_BASE_URL must be a valid absolute URL');
    });

    it('parses previous jwt secrets for secret rotation', () => {
        const previousSecret = 'p'.repeat(32);
        const secondPreviousSecret = 'q'.repeat(32);
        const environment = validateEnvironment({
            ...validEnvironment,
            JWT_PREVIOUS_SECRETS: `${previousSecret}, ${secondPreviousSecret}, ${previousSecret}`,
        });

        expect(environment.JWT_PREVIOUS_SECRETS).toEqual([
            previousSecret,
            secondPreviousSecret,
        ]);
    });

    it('enables csrf origin checks by default only in production', () => {
        expect(
            validateEnvironment({
                ...validEnvironment,
                NODE_ENV: 'development',
            }).CSRF_ORIGIN_CHECK_ENABLED,
        ).toBe(false);

        expect(
            validateEnvironment({
                ...validEnvironment,
                NODE_ENV: 'production',
            }).CSRF_ORIGIN_CHECK_ENABLED,
        ).toBe(true);
    });

    it('validates TTL ordering correctly', () => {
        // Valid order: access (15m) <= refresh (7d) <= session (7d)
        expect(() => validateEnvironment(validEnvironment)).not.toThrow();

        // Invalid: access (8d) > refresh (7d)
        const errorAccess = getValidationError({
            ...validEnvironment,
            JWT_ACCESS_TOKEN_TTL: '8d',
            JWT_REFRESH_TOKEN_TTL: '7d',
        });
        expect(errorAccess.message).toContain(
            'JWT_ACCESS_TOKEN_TTL must be less than or equal to JWT_REFRESH_TOKEN_TTL',
        );

        // Invalid: refresh (8d) > session (7d)
        const errorRefresh = getValidationError({
            ...validEnvironment,
            JWT_REFRESH_TOKEN_TTL: '8d',
            SESSION_MAX_TTL: '7d',
        });
        expect(errorRefresh.message).toContain(
            'JWT_REFRESH_TOKEN_TTL must be less than or equal to SESSION_MAX_TTL',
        );
    });

    it('parses rollout flags correctly when configured', () => {
        const environment = validateEnvironment({
            ...validEnvironment,
            DEVICE_ID_FORWARDING_ENABLED: 'true',
            REFRESH_ROTATION_ENABLED: 'true',
            DEVICE_BINDING_REQUIRED_CUSTOMER: 'false',
            DEVICE_BINDING_REQUIRED_ADMIN: 'true',
            LOGIN_CHECKPOINT_ENABLED: 'true',
            REAUTH_ENFORCEMENT_ENABLED: 'true',
        });

        expect(environment.DEVICE_ID_FORWARDING_ENABLED).toBe(true);
        expect(environment.REFRESH_ROTATION_ENABLED).toBe(true);
        expect(environment.DEVICE_BINDING_REQUIRED_CUSTOMER).toBe(false);
        expect(environment.DEVICE_BINDING_REQUIRED_ADMIN).toBe(true);
        expect(environment.LOGIN_CHECKPOINT_ENABLED).toBe(true);
        expect(environment.REAUTH_ENFORCEMENT_ENABLED).toBe(true);
    });

    it('reports all missing core variables in one error', () => {
        const error = getValidationError({});

        expect(error.message).toContain('COOKIE_DOMAIN is required');
        expect(error.message).toContain('DB_HOST is required');
        expect(error.message).toContain('DB_PORT is required');
        expect(error.message).toContain('JWT_SECRET is required');
        expect(error.message).toContain('ENCRYPTION_KEY is required');
    });

    it('rejects invalid ports, URLs, durations, booleans, and constrained values', () => {
        const error = getValidationError({
            ...validEnvironment,
            NODE_ENV: 'staging',
            PORT: '70000',
            CLIENT_ORIGIN: 'localhost:3001',
            JWT_ACCESS_TOKEN_TTL: 'later',
            DEVICE_ID_COOKIE_TTL: 'later',
            MAIL_RETRY_BACKOFF_TYPE: 'linear',
            MAIL_RETRY_BACKOFF_JITTER: '1.5',
            DEVICE_ID_FORWARDING_ENABLED: 'maybe',
            REFRESH_ROTATION_ENABLED: 'maybe',
            DEVICE_BINDING_REQUIRED_CUSTOMER: 'maybe',
            DEVICE_BINDING_REQUIRED_ADMIN: 'maybe',
            LOGIN_CHECKPOINT_ENABLED: 'maybe',
            REAUTH_ENFORCEMENT_ENABLED: 'maybe',
            CSRF_ORIGIN_CHECK_ENABLED: 'maybe',
            LOG_LEVEL: 'loud',
            LOG_FORMAT: 'xml',
            LOG_OUTPUT: 'database',
            LOG_FILE_MAX_DAYS: '0',
            LOG_REDACTION_ENABLED: 'maybe',
            LOG_REQUEST_ID_HEADER: 'invalid header',
        });

        expect(error.message).toContain('Environment validation failed');
        expect(error.message).toContain('NODE_ENV must be one of');
        expect(error.message).toContain('PORT must be less than or equal to 65535');
        expect(error.message).toContain('CLIENT_ORIGIN must use http or https');
        expect(error.message).toContain('JWT_ACCESS_TOKEN_TTL must be a positive duration');
        expect(error.message).toContain('DEVICE_ID_COOKIE_TTL must be a positive duration');
        expect(error.message).toContain('MAIL_RETRY_BACKOFF_TYPE must be one of');
        expect(error.message).toContain(
            'MAIL_RETRY_BACKOFF_JITTER must be less than or equal to 1',
        );
        expect(error.message).toContain('DEVICE_ID_FORWARDING_ENABLED must be either');
        expect(error.message).toContain('REFRESH_ROTATION_ENABLED must be either');
        expect(error.message).toContain('DEVICE_BINDING_REQUIRED_CUSTOMER must be either');
        expect(error.message).toContain('DEVICE_BINDING_REQUIRED_ADMIN must be either');
        expect(error.message).toContain('LOGIN_CHECKPOINT_ENABLED must be either');
        expect(error.message).toContain('REAUTH_ENFORCEMENT_ENABLED must be either');
        expect(error.message).toContain('CSRF_ORIGIN_CHECK_ENABLED must be either');
        expect(error.message).toContain('LOG_LEVEL must be one of');
        expect(error.message).toContain('LOG_FORMAT must be one of');
        expect(error.message).toContain('LOG_OUTPUT must be one of');
        expect(error.message).toContain('LOG_FILE_MAX_DAYS must be greater than or equal to 1');
        expect(error.message).toContain('LOG_REDACTION_ENABLED must be either');
        expect(error.message).toContain('LOG_REQUEST_ID_HEADER must be a valid HTTP header name');
    });

    it('rejects pretty formatting for file output', () => {
        const error = getValidationError({
            ...validEnvironment,
            LOG_OUTPUT: 'file',
            LOG_FORMAT: 'pretty',
        });

        expect(error.message).toContain(
            'LOG_FORMAT=pretty is only supported with LOG_OUTPUT=console',
        );
    });

    it('requires strong core secrets and validates an optional mail encryption key', () => {
        const error = getValidationError({
            ...validEnvironment,
            JWT_SECRET: 'short',
            ENCRYPTION_KEY: 'short',
            MAIL_SETTINGS_ENCRYPTION_KEY: 'short',
            JWT_PREVIOUS_SECRETS: 'short',
        });

        expect(error.message).toContain('JWT_SECRET must contain at least 32 characters');
        expect(error.message).toContain('ENCRYPTION_KEY must contain at least 32 characters');
        expect(error.message).toContain(
            'MAIL_SETTINGS_ENCRYPTION_KEY must contain at least 32 characters when configured',
        );
        expect(error.message).toContain(
            'JWT_PREVIOUS_SECRETS entries must contain at least 32 characters',
        );
    });

    it('individually rejects invalid values for each rollout flag', () => {
        const flags = [
            'DEVICE_ID_FORWARDING_ENABLED',
            'REFRESH_ROTATION_ENABLED',
            'DEVICE_BINDING_REQUIRED_CUSTOMER',
            'DEVICE_BINDING_REQUIRED_ADMIN',
            'LOGIN_CHECKPOINT_ENABLED',
            'REAUTH_ENFORCEMENT_ENABLED',
            'CSRF_ORIGIN_CHECK_ENABLED',
        ];

        for (const flag of flags) {
            const error = getValidationError({
                ...validEnvironment,
                [flag]: 'maybe',
            });
            expect(error.message).toContain(`${flag} must be either "true" or "false"`);
        }
    });

    it('rejects permissive trusted proxy settings in production', () => {
        const trueError = getValidationError({
            ...validEnvironment,
            NODE_ENV: 'production',
            TRUST_PROXY: 'true',
        });
        expect(trueError.message).toContain('TRUST_PROXY=true is not allowed in production');

        const hopCountError = getValidationError({
            ...validEnvironment,
            NODE_ENV: 'production',
            TRUST_PROXY: '1',
        });
        expect(hopCountError.message).toContain(
            'Numeric TRUST_PROXY hop counts are not allowed in production',
        );

        const namedRangeError = getValidationError({
            ...validEnvironment,
            NODE_ENV: 'production',
            TRUST_PROXY: 'loopback',
        });
        expect(namedRangeError.message).toContain(
            'Named TRUST_PROXY ranges are not allowed in production',
        );
    });

    it('accepts explicit trusted proxy CIDR ranges', () => {
        const environment = validateEnvironment({
            ...validEnvironment,
            NODE_ENV: 'production',
            TRUST_PROXY: '10.0.0.0/8, 192.168.0.0/16',
        });

        expect(environment.TRUST_PROXY).toBe('10.0.0.0/8, 192.168.0.0/16');
    });
});
