import ms, { type StringValue } from 'ms';
import { INITIAL_CAPACITY_LIMITS } from '@/config/capacity-limits.config';
import { isIP } from 'net';
import {
    DEFAULT_DEVELOPMENT_LOGGER_CONFIG,
    DEFAULT_LOG_FILE_DIR,
    DEFAULT_LOG_FILE_MAX_DAYS,
    DEFAULT_LOG_REQUEST_ID_HEADER,
    DEFAULT_PRODUCTION_LOGGER_CONFIG,
    DEFAULT_SERVICE_NAME,
    LOGGER_FORMAT_VALUES,
    LOGGER_LEVEL_VALUES,
    LOGGER_OUTPUT_VALUES,
} from '@/module-logger/constants/logger.constants';
import type {
    LoggerFormat,
    LoggerLevel,
    LoggerOutput,
} from '@/module-logger/interfaces/logger.interfaces';

export interface Environment {
    NODE_ENV: 'development' | 'test' | 'production';
    PORT: number;
    CLIENT_ORIGIN: string;
    CLIENT_PUBLIC_URL: string;
    COOKIE_DOMAIN: string;
    DB_HOST: string;
    DB_PORT: number;
    DB_USER: string;
    DB_PASSWORD: string;
    DB_NAME: string;
    DB_POOL_MAX: number;
    REDIS_HOST: string;
    REDIS_PORT: number;
    REDIS_DB: number;
    REDIS_PASSWORD?: string;
    JWT_SECRET: string;
    JWT_PREVIOUS_SECRETS: string[];
    JWT_ACCESS_TOKEN_TTL: string;
    JWT_REFRESH_TOKEN_TTL: string;
    SESSION_MAX_TTL: string;
    DEVICE_ID_COOKIE_TTL: string;
    AUTH_CLOCK_SKEW_SECONDS: number;
    ENCRYPTION_KEY: string;
    MAIL_SETTINGS_ENCRYPTION_KEY?: string;
    MAIL_QUEUE_LIMITER_MAX: number;
    MAIL_QUEUE_LIMITER_DURATION: number;
    MAIL_WORKER_CONCURRENCY: number;
    MAIL_RETRY_ATTEMPTS: number;
    MAIL_RETRY_BACKOFF_DELAY: number;
    MAIL_RETRY_BACKOFF_TYPE: 'exponential' | 'fixed';
    MAIL_RETRY_BACKOFF_JITTER: number;
    DEVICE_ID_FORWARDING_ENABLED: boolean;
    REFRESH_ROTATION_ENABLED: boolean;
    DEVICE_BINDING_REQUIRED_CUSTOMER: boolean;
    DEVICE_BINDING_REQUIRED_ADMIN: boolean;
    LOGIN_CHECKPOINT_ENABLED: boolean;
    REAUTH_ENFORCEMENT_ENABLED: boolean;
    CSRF_ORIGIN_CHECK_ENABLED: boolean;
    TRUST_PROXY?: string;
    GEOIP_TIMEOUT_MS: number;
    SERVICE_TO_SERVICE_SECRET?: string;
    TRADING_ONBOARDING_TOKEN_SECRET?: string;
    TRADING_ONBOARDING_TOKEN_TTL_SECONDS: number;
    BOT_PLATFORM_BASE_URL?: string;
    LOG_LEVEL: LoggerLevel;
    LOG_FORMAT: LoggerFormat;
    LOG_OUTPUT: LoggerOutput;
    LOG_FILE_DIR: string;
    LOG_FILE_MAX_DAYS: number;
    LOG_REDACTION_ENABLED: boolean;
    LOG_STRUCTURED_BOOTSTRAP_ENABLED: boolean;
    LOG_REQUEST_ID_HEADER: string;
    SERVICE_NAME: string;
}

export interface LoggerEnvironment {
    NODE_ENV: Environment['NODE_ENV'];
    LOG_LEVEL: LoggerLevel;
    LOG_FORMAT: LoggerFormat;
    LOG_OUTPUT: LoggerOutput;
    LOG_FILE_DIR: string;
    LOG_FILE_MAX_DAYS: number;
    LOG_REDACTION_ENABLED: boolean;
    LOG_STRUCTURED_BOOTSTRAP_ENABLED: boolean;
    LOG_REQUEST_ID_HEADER: string;
    SERVICE_NAME: string;
}

const NODE_ENV_VALUES = new Set(['development', 'test', 'production']);
const MAIL_RETRY_BACKOFF_TYPES = new Set(['exponential', 'fixed']);
type NodeEnvironment = Environment['NODE_ENV'];
type MailRetryBackoffType = Environment['MAIL_RETRY_BACKOFF_TYPE'];

export function validateEnvironment(input: Record<string, unknown>): Environment {
    const environment: Record<string, unknown> = { ...input };
    const errors: string[] = [];

    environment.NODE_ENV = readEnum<NodeEnvironment>(
        environment,
        'NODE_ENV',
        NODE_ENV_VALUES,
        errors,
        'development',
    );
    const nodeEnv = environment.NODE_ENV as NodeEnvironment;
    environment.PORT = readInteger(environment, 'PORT', errors, {
        defaultValue: 3000,
        min: 1,
        max: 65535,
    });
    environment.CLIENT_ORIGIN = readHttpUrl(environment, 'CLIENT_ORIGIN', errors, {
        defaultValue: 'http://localhost:3001',
    });
    environment.CLIENT_PUBLIC_URL = readHttpUrl(environment, 'CLIENT_PUBLIC_URL', errors, {
        defaultValue: 'https://localhost',
    });

    environment.COOKIE_DOMAIN = readRequiredString(environment, 'COOKIE_DOMAIN', errors);

    environment.DB_HOST = readRequiredString(environment, 'DB_HOST', errors);
    environment.DB_PORT = readInteger(environment, 'DB_PORT', errors, {
        required: true,
        min: 1,
        max: 65535,
    });
    environment.DB_USER = readRequiredString(environment, 'DB_USER', errors);
    environment.DB_PASSWORD = readRequiredString(environment, 'DB_PASSWORD', errors);
    environment.DB_NAME = readRequiredString(environment, 'DB_NAME', errors);
    environment.DB_POOL_MAX = readInteger(environment, 'DB_POOL_MAX', errors, {
        defaultValue: INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica,
        min: 1,
        max: INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica,
    });

    environment.REDIS_HOST = readString(environment, 'REDIS_HOST') ?? 'localhost';
    environment.REDIS_PORT = readInteger(environment, 'REDIS_PORT', errors, {
        defaultValue: 6379,
        min: 1,
        max: 65535,
    });
    environment.REDIS_DB = readInteger(environment, 'REDIS_DB', errors, {
        defaultValue: 0,
        min: 0,
    });
    environment.REDIS_PASSWORD = readOptionalString(environment, 'REDIS_PASSWORD');

    environment.JWT_SECRET = readSecret(environment, 'JWT_SECRET', errors);
    environment.JWT_PREVIOUS_SECRETS = readSecretList(environment, 'JWT_PREVIOUS_SECRETS', errors);
    environment.JWT_ACCESS_TOKEN_TTL = readDuration(environment, 'JWT_ACCESS_TOKEN_TTL', errors);
    environment.JWT_REFRESH_TOKEN_TTL = readDuration(environment, 'JWT_REFRESH_TOKEN_TTL', errors);
    environment.SESSION_MAX_TTL = readDuration(environment, 'SESSION_MAX_TTL', errors);
    environment.DEVICE_ID_COOKIE_TTL = readDuration(environment, 'DEVICE_ID_COOKIE_TTL', errors);
    environment.AUTH_CLOCK_SKEW_SECONDS = readInteger(
        environment,
        'AUTH_CLOCK_SKEW_SECONDS',
        errors,
        {
            defaultValue: 30,
            min: 0,
        },
    );

    const accessTokenTtlMs = environment.JWT_ACCESS_TOKEN_TTL
        ? ms(environment.JWT_ACCESS_TOKEN_TTL as StringValue)
        : undefined;
    const refreshTokenTtlMs = environment.JWT_REFRESH_TOKEN_TTL
        ? ms(environment.JWT_REFRESH_TOKEN_TTL as StringValue)
        : undefined;
    const sessionMaxTtlMs = environment.SESSION_MAX_TTL
        ? ms(environment.SESSION_MAX_TTL as StringValue)
        : undefined;

    if (
        typeof accessTokenTtlMs === 'number' &&
        Number.isFinite(accessTokenTtlMs) &&
        typeof refreshTokenTtlMs === 'number' &&
        Number.isFinite(refreshTokenTtlMs) &&
        typeof sessionMaxTtlMs === 'number' &&
        Number.isFinite(sessionMaxTtlMs)
    ) {
        if (accessTokenTtlMs > refreshTokenTtlMs) {
            errors.push('JWT_ACCESS_TOKEN_TTL must be less than or equal to JWT_REFRESH_TOKEN_TTL');
        }
        if (refreshTokenTtlMs > sessionMaxTtlMs) {
            errors.push('JWT_REFRESH_TOKEN_TTL must be less than or equal to SESSION_MAX_TTL');
        }
    }

    environment.ENCRYPTION_KEY = readSecret(environment, 'ENCRYPTION_KEY', errors);
    environment.MAIL_SETTINGS_ENCRYPTION_KEY = readOptionalSecret(
        environment,
        'MAIL_SETTINGS_ENCRYPTION_KEY',
        errors,
    );

    environment.MAIL_QUEUE_LIMITER_MAX = readInteger(
        environment,
        'MAIL_QUEUE_LIMITER_MAX',
        errors,
        {
            defaultValue: INITIAL_CAPACITY_LIMITS.criticalEmailRequestsPerSecond,
            min: 1,
        },
    );
    environment.MAIL_QUEUE_LIMITER_DURATION = readInteger(
        environment,
        'MAIL_QUEUE_LIMITER_DURATION',
        errors,
        {
            defaultValue: INITIAL_CAPACITY_LIMITS.mailQueueLimiterDurationMs,
            min: 1,
        },
    );
    environment.MAIL_WORKER_CONCURRENCY = readInteger(
        environment,
        'MAIL_WORKER_CONCURRENCY',
        errors,
        {
            defaultValue: INITIAL_CAPACITY_LIMITS.mailWorkerConcurrencyPerApiReplica,
            min: 1,
        },
    );
    environment.MAIL_RETRY_ATTEMPTS = readInteger(environment, 'MAIL_RETRY_ATTEMPTS', errors, {
        defaultValue: 3,
        min: 1,
    });
    environment.MAIL_RETRY_BACKOFF_DELAY = readInteger(
        environment,
        'MAIL_RETRY_BACKOFF_DELAY',
        errors,
        {
            defaultValue: 2000,
            min: 1,
        },
    );
    environment.MAIL_RETRY_BACKOFF_TYPE = readEnum<MailRetryBackoffType>(
        environment,
        'MAIL_RETRY_BACKOFF_TYPE',
        MAIL_RETRY_BACKOFF_TYPES,
        errors,
        'exponential',
    );
    environment.MAIL_RETRY_BACKOFF_JITTER = readNumber(
        environment,
        'MAIL_RETRY_BACKOFF_JITTER',
        errors,
        {
            defaultValue: 0.2,
            min: 0,
            max: 1,
        },
    );

    environment.DEVICE_ID_FORWARDING_ENABLED = readBoolean(
        environment,
        'DEVICE_ID_FORWARDING_ENABLED',
        errors,
        false,
    );
    environment.REFRESH_ROTATION_ENABLED = readBoolean(
        environment,
        'REFRESH_ROTATION_ENABLED',
        errors,
        false,
    );
    environment.DEVICE_BINDING_REQUIRED_CUSTOMER = readBoolean(
        environment,
        'DEVICE_BINDING_REQUIRED_CUSTOMER',
        errors,
        false,
    );
    environment.DEVICE_BINDING_REQUIRED_ADMIN = readBoolean(
        environment,
        'DEVICE_BINDING_REQUIRED_ADMIN',
        errors,
        false,
    );
    environment.LOGIN_CHECKPOINT_ENABLED = readBoolean(
        environment,
        'LOGIN_CHECKPOINT_ENABLED',
        errors,
        false,
    );

    environment.REAUTH_ENFORCEMENT_ENABLED = readBoolean(
        environment,
        'REAUTH_ENFORCEMENT_ENABLED',
        errors,
        false,
    );
    environment.CSRF_ORIGIN_CHECK_ENABLED = readBoolean(
        environment,
        'CSRF_ORIGIN_CHECK_ENABLED',
        errors,
        nodeEnv === 'production',
    );

    environment.TRUST_PROXY = readTrustProxy(environment, errors, nodeEnv);

    environment.GEOIP_TIMEOUT_MS = readInteger(environment, 'GEOIP_TIMEOUT_MS', errors, {
        required: false,
        defaultValue: 1000,
        min: 1,
    });
    environment.SERVICE_TO_SERVICE_SECRET = readOptionalSecret(
        environment,
        'SERVICE_TO_SERVICE_SECRET',
        errors,
    );
    environment.TRADING_ONBOARDING_TOKEN_SECRET = readOptionalSecret(
        environment,
        'TRADING_ONBOARDING_TOKEN_SECRET',
        errors,
    );
    environment.TRADING_ONBOARDING_TOKEN_TTL_SECONDS = readInteger(
        environment,
        'TRADING_ONBOARDING_TOKEN_TTL_SECONDS',
        errors,
        {
            defaultValue: 300,
            min: 30,
            max: 3600,
        },
    );
    environment.BOT_PLATFORM_BASE_URL = readOptionalHttpUrl(
        environment,
        'BOT_PLATFORM_BASE_URL',
        errors,
    );

    Object.assign(environment, readLoggerEnvironment(environment, errors, nodeEnv));

    if (errors.length > 0) {
        throw new Error(`Environment validation failed:\n- ${errors.join('\n- ')}`);
    }

    return environment as unknown as Environment;
}

export function validateLoggerEnvironment(input: Record<string, unknown>): LoggerEnvironment {
    const environment: Record<string, unknown> = { ...input };
    const errors: string[] = [];
    const nodeEnv = readEnum<NodeEnvironment>(
        environment,
        'NODE_ENV',
        NODE_ENV_VALUES,
        errors,
        'development',
    );

    const loggerEnvironment = readLoggerEnvironment(environment, errors, nodeEnv);

    if (errors.length > 0) {
        throw new Error(`Logger environment validation failed:\n- ${errors.join('\n- ')}`);
    }

    return {
        NODE_ENV: nodeEnv,
        ...loggerEnvironment,
    };
}

function readLoggerEnvironment(
    environment: Record<string, unknown>,
    errors: string[],
    nodeEnv: NodeEnvironment,
): Omit<LoggerEnvironment, 'NODE_ENV'> {
    const defaults =
        nodeEnv === 'production'
            ? DEFAULT_PRODUCTION_LOGGER_CONFIG
            : DEFAULT_DEVELOPMENT_LOGGER_CONFIG;

    const requestIdHeader =
        readString(environment, 'LOG_REQUEST_ID_HEADER') ?? DEFAULT_LOG_REQUEST_ID_HEADER;

    if (!/^[a-z0-9!#$%&'*+.^_`|~-]+$/i.test(requestIdHeader)) {
        errors.push('LOG_REQUEST_ID_HEADER must be a valid HTTP header name');
    }

    const logOutput = readEnum<LoggerOutput>(
        environment,
        'LOG_OUTPUT',
        LOGGER_OUTPUT_VALUES,
        errors,
        defaults.output,
    );
    const logFormat = readEnum<LoggerFormat>(
        environment,
        'LOG_FORMAT',
        LOGGER_FORMAT_VALUES,
        errors,
        defaults.format,
    );

    if (logOutput === 'file' && logFormat === 'pretty') {
        errors.push('LOG_FORMAT=pretty is only supported with LOG_OUTPUT=console');
    }

    return {
        LOG_LEVEL: readEnum<LoggerLevel>(
            environment,
            'LOG_LEVEL',
            LOGGER_LEVEL_VALUES,
            errors,
            defaults.level,
        ),
        LOG_FORMAT: logFormat,
        LOG_OUTPUT: logOutput,
        LOG_FILE_DIR: readString(environment, 'LOG_FILE_DIR') ?? DEFAULT_LOG_FILE_DIR,
        LOG_FILE_MAX_DAYS: readInteger(environment, 'LOG_FILE_MAX_DAYS', errors, {
            defaultValue: DEFAULT_LOG_FILE_MAX_DAYS,
            min: 1,
            max: 3650,
        }),
        LOG_REDACTION_ENABLED: readBoolean(
            environment,
            'LOG_REDACTION_ENABLED',
            errors,
            defaults.redactionEnabled,
        ),
        LOG_STRUCTURED_BOOTSTRAP_ENABLED: readBoolean(
            environment,
            'LOG_STRUCTURED_BOOTSTRAP_ENABLED',
            errors,
            defaults.structuredBootstrapEnabled,
        ),
        LOG_REQUEST_ID_HEADER: requestIdHeader,
        SERVICE_NAME: readString(environment, 'SERVICE_NAME') ?? DEFAULT_SERVICE_NAME,
    };
}

function readRequiredString(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
): string {
    const value = readString(environment, key);

    if (value === undefined) {
        errors.push(`${key} is required`);
        return '';
    }

    return value;
}

function readString(environment: Record<string, unknown>, key: string): string | undefined {
    const rawValue = environment[key];

    if (rawValue === undefined || rawValue === null) {
        return undefined;
    }

    if (
        typeof rawValue !== 'string' &&
        typeof rawValue !== 'number' &&
        typeof rawValue !== 'boolean'
    ) {
        return undefined;
    }

    const value = String(rawValue).trim();
    return value.length > 0 ? value : undefined;
}

function readOptionalString(environment: Record<string, unknown>, key: string): string | undefined {
    return readString(environment, key);
}

function readTrustProxy(
    environment: Record<string, unknown>,
    errors: string[],
    nodeEnv: Environment['NODE_ENV'],
): string {
    const value = readString(environment, 'TRUST_PROXY') ?? 'false';
    const normalized = value.toLowerCase();

    if (normalized === 'true' && nodeEnv === 'production') {
        errors.push(
            'TRUST_PROXY=true is not allowed in production; configure explicit trusted proxy IP/CIDR ranges or set TRUST_PROXY=false',
        );
    }

    if (/^\d+$/.test(value) && nodeEnv === 'production') {
        errors.push(
            'Numeric TRUST_PROXY hop counts are not allowed in production; configure explicit trusted proxy IP/CIDR ranges or set TRUST_PROXY=false',
        );
    }

    if (nodeEnv === 'production' && containsNamedTrustProxyRange(value)) {
        errors.push(
            'Named TRUST_PROXY ranges are not allowed in production; configure explicit trusted proxy IP/CIDR ranges or set TRUST_PROXY=false',
        );
    }

    if (!isValidTrustProxy(value)) {
        errors.push(
            'TRUST_PROXY must be "true", "false", a hop count, or a comma-separated list of trusted proxy IP/CIDR ranges',
        );
    }

    return value;
}

function isValidTrustProxy(value: string): boolean {
    const normalized = value.toLowerCase();
    if (normalized === 'true' || normalized === 'false' || /^\d+$/.test(value)) {
        return true;
    }

    return value.split(',').every(rawEntry => {
        const entry = rawEntry.trim();
        if (!entry) {
            return false;
        }

        if (['loopback', 'linklocal', 'uniquelocal'].includes(entry.toLowerCase())) {
            return true;
        }

        const [address, prefix] = entry.split('/');
        if (!address || isIP(address) === 0) {
            return false;
        }

        if (prefix === undefined) {
            return true;
        }

        if (!/^\d+$/.test(prefix)) {
            return false;
        }

        const prefixLength = Number(prefix);
        return prefixLength >= 0 && prefixLength <= (isIP(address) === 4 ? 32 : 128);
    });
}

function containsNamedTrustProxyRange(value: string): boolean {
    const namedRanges = new Set(['loopback', 'linklocal', 'uniquelocal']);

    return value.split(',').some(entry => namedRanges.has(entry.trim().toLowerCase()));
}

function readSecret(environment: Record<string, unknown>, key: string, errors: string[]): string {
    const value = readRequiredString(environment, key, errors);

    if (value && value.length < 32) {
        errors.push(`${key} must contain at least 32 characters`);
    }

    return value;
}

function readOptionalSecret(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
): string | undefined {
    const value = readString(environment, key);

    if (value !== undefined && value.length < 32) {
        errors.push(`${key} must contain at least 32 characters when configured`);
    }

    return value;
}

function readSecretList(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
): string[] {
    const value = readString(environment, key);

    if (value === undefined) {
        return [];
    }

    const secrets = value
        .split(',')
        .map(secret => secret.trim())
        .filter(Boolean);

    for (const secret of secrets) {
        if (secret.length < 32) {
            errors.push(`${key} entries must contain at least 32 characters`);
            break;
        }
    }

    return [...new Set(secrets)];
}

function readDuration(environment: Record<string, unknown>, key: string, errors: string[]): string {
    const value = readRequiredString(environment, key, errors);

    if (!value) {
        return value;
    }

    const duration = ms(value as StringValue);

    if (!Number.isFinite(duration) || duration <= 0) {
        errors.push(`${key} must be a positive duration such as "15m" or "7d"`);
    }

    return value;
}

function readHttpUrl(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
    options: { defaultValue?: string } = {},
): string {
    const value = readString(environment, key) ?? options.defaultValue;

    if (!value) {
        errors.push(`${key} is required`);
        return '';
    }

    try {
        const url = new URL(value);

        if (url.protocol !== 'http:' && url.protocol !== 'https:') {
            errors.push(`${key} must use http or https`);
        }
    } catch {
        errors.push(`${key} must be a valid absolute URL`);
    }

    return value;
}

function readOptionalHttpUrl(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
): string | undefined {
    const value = readString(environment, key);

    if (!value) {
        return undefined;
    }

    try {
        const url = new URL(value);

        if (url.protocol !== 'http:' && url.protocol !== 'https:') {
            errors.push(`${key} must use http or https`);
        }
    } catch {
        errors.push(`${key} must be a valid absolute URL`);
    }

    return value;
}

function readBoolean(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
    defaultValue: boolean,
): boolean {
    const value = readString(environment, key);

    if (value === undefined) {
        return defaultValue;
    }

    if (value === 'true') {
        return true;
    }

    if (value === 'false') {
        return false;
    }

    errors.push(`${key} must be either "true" or "false"`);
    return defaultValue;
}

function readEnum<T extends string>(
    environment: Record<string, unknown>,
    key: string,
    allowedValues: Set<string>,
    errors: string[],
    defaultValue: T,
): T {
    const value = readString(environment, key) ?? defaultValue;

    if (!allowedValues.has(value)) {
        errors.push(`${key} must be one of: ${[...allowedValues].join(', ')}`);
        return defaultValue;
    }

    return value as T;
}

function readInteger(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
    options: {
        required?: boolean;
        defaultValue?: number;
        min?: number;
        max?: number;
    },
): number {
    const value = readNumber(environment, key, errors, options);

    if (!Number.isFinite(value)) {
        return value;
    }

    if (!Number.isInteger(value)) {
        errors.push(`${key} must be an integer`);
    }

    return value;
}

function readNumber(
    environment: Record<string, unknown>,
    key: string,
    errors: string[],
    options: {
        required?: boolean;
        defaultValue?: number;
        min?: number;
        max?: number;
    },
): number {
    const rawValue = readString(environment, key);

    if (rawValue === undefined) {
        if (options.defaultValue !== undefined) {
            return options.defaultValue;
        }

        if (options.required) {
            errors.push(`${key} is required`);
        }

        return Number.NaN;
    }

    const value = Number(rawValue);

    if (!Number.isFinite(value)) {
        errors.push(`${key} must be a finite number`);
        return value;
    }

    if (options.min !== undefined && value < options.min) {
        errors.push(`${key} must be greater than or equal to ${options.min}`);
    }

    if (options.max !== undefined && value > options.max) {
        errors.push(`${key} must be less than or equal to ${options.max}`);
    }

    return value;
}
