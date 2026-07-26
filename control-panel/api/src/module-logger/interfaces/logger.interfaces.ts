import type { LoggerService } from '@nestjs/common';

export type LoggerEnvironmentName = 'development' | 'test' | 'production';
export type LoggerLevel = 'fatal' | 'error' | 'warn' | 'info' | 'debug' | 'trace' | 'silent';
export type LoggerFormat = 'json' | 'pretty';
export type LoggerOutput = 'console' | 'file';
export type LoggerChannel =
    | 'system'
    | 'auth'
    | 'auth-token'
    | 'mail'
    | 'data-patch';

export interface LoggerConfig {
    environment: LoggerEnvironmentName;
    format: LoggerFormat;
    level: LoggerLevel;
    output: LoggerOutput;
    fileDir: string;
    fileMaxDays: number;
    redactionEnabled: boolean;
    structuredBootstrapEnabled: boolean;
    requestIdHeader: string;
    serviceName: string;
}

export interface NormalizedLogPayload {
    context?: string;
    error?: SerializedError;
    metadata?: unknown;
    requestId?: string;
    stack?: string;
}

export interface SerializedError {
    name: string;
    message: string;
    stack?: string;
    cause?: unknown;
}

export interface RequestLogContext {
    requestId: string;
}

export type RuntimeLogger = LoggerService;
