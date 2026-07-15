import type {
    LoggerConfig,
    LoggerChannel,
    LoggerFormat,
    LoggerLevel,
    LoggerOutput,
} from '@/module-logger/interfaces/logger.interfaces';

export const LOGGER_LEVEL_VALUES = new Set<LoggerLevel>([
    'fatal',
    'error',
    'warn',
    'info',
    'debug',
    'trace',
    'silent',
]);

export const LOGGER_FORMAT_VALUES = new Set<LoggerFormat>(['json', 'pretty']);

export const LOGGER_OUTPUT_VALUES = new Set<LoggerOutput>(['console', 'file']);

export const LOGGER_CHANNELS = [
    'system',
    'auth',
    'auth-token',
    'mail',
    'data-patch',
] as const satisfies readonly LoggerChannel[];

export const DEFAULT_SERVICE_NAME = 'pampilo-api';
export const DEFAULT_LOG_REQUEST_ID_HEADER = 'x-request-id';
export const DEFAULT_LOG_FILE_DIR = 'logs';
export const DEFAULT_LOG_FILE_MAX_DAYS = 14;
export const LOG_REDACTION_CENSOR = '[REDACTED]';

export const LOGGER_REDACTED_KEYS = new Set([
    'authorization',
    'cookie',
    'setcookie',
    'password',
    'passwordhash',
    'token',
    'accesstoken',
    'refreshtoken',
    'jwt',
    'secret',
    'clientsecret',
    'apikey',
    'key',
    'encryptionkey',
    'smtppassword',
    'servicetoken',
    'csrf',
    'sessionid',
]);

export const PINO_REDACT_PATHS = [
    'req.headers.authorization',
    'req.headers.cookie',
    'res.headers.set-cookie',
    'headers.authorization',
    'headers.cookie',
    'headers.set-cookie',
    '*.authorization',
    '*.cookie',
    '*.set-cookie',
    '*.password',
    '*.passwordHash',
    '*.token',
    '*.accessToken',
    '*.refreshToken',
    '*.jwt',
    '*.secret',
    '*.clientSecret',
    '*.apiKey',
    '*.encryptionKey',
    '*.smtpPassword',
] as const;

export const DEFAULT_PRODUCTION_LOGGER_CONFIG: LoggerConfig = {
    environment: 'production',
    format: 'json',
    level: 'info',
    output: 'console',
    fileDir: DEFAULT_LOG_FILE_DIR,
    fileMaxDays: DEFAULT_LOG_FILE_MAX_DAYS,
    redactionEnabled: true,
    structuredBootstrapEnabled: true,
    requestIdHeader: DEFAULT_LOG_REQUEST_ID_HEADER,
    serviceName: DEFAULT_SERVICE_NAME,
};

export const DEFAULT_DEVELOPMENT_LOGGER_CONFIG: LoggerConfig = {
    environment: 'development',
    format: 'json',
    level: 'debug',
    output: 'console',
    fileDir: DEFAULT_LOG_FILE_DIR,
    fileMaxDays: DEFAULT_LOG_FILE_MAX_DAYS,
    redactionEnabled: true,
    structuredBootstrapEnabled: false,
    requestIdHeader: DEFAULT_LOG_REQUEST_ID_HEADER,
    serviceName: DEFAULT_SERVICE_NAME,
};
