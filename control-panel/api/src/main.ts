import { NestFactory } from '@nestjs/core';
import { AppModule } from '@/app.module';
import { ValidationPipe, type NestApplicationOptions } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { setupSwagger } from '@/common/utils/setup-swagger.util';
import cookieParser from 'cookie-parser';
import { json, static as expressStatic, urlencoded } from 'express';
import type { Application } from 'express';
import { join } from 'path';
import {
    createBootstrapLoggerConfigFromEnv,
    createStructuredBootstrapLoggerFromEnv,
} from '@/module-logger/utils/logger-env.util';
import { createRequestIdMiddleware } from '@/module-logger/middleware/request-id.middleware';
import { createSecurityHeadersMiddleware } from '@/common/security/security-headers.middleware';
import { createCsrfOriginMiddleware } from '@/common/security/csrf-origin.middleware';

async function bootstrap(): Promise<void> {
    const loggerConfig = createBootstrapLoggerConfigFromEnv();
    const nestFactoryOptions: NestApplicationOptions = {
        bodyParser: false,
    };

    if (loggerConfig.environment === 'production' || loggerConfig.structuredBootstrapEnabled) {
        const structuredLogger = createStructuredBootstrapLoggerFromEnv(loggerConfig);

        if (structuredLogger) {
            nestFactoryOptions.logger = structuredLogger;
        }
    }

    const app = await NestFactory.create(AppModule, nestFactoryOptions);
    const configService = app.get(ConfigService);
    const expressApp = app.getHttpAdapter().getInstance() as Application;
    const trustProxy = configService.get<string>('TRUST_PROXY');
    if (trustProxy === 'true') {
        expressApp.set('trust proxy', true);
    } else if (trustProxy === 'false') {
        expressApp.set('trust proxy', false);
    } else if (trustProxy && /^\d+$/.test(trustProxy)) {
        expressApp.set('trust proxy', parseInt(trustProxy, 10));
    } else if (trustProxy) {
        expressApp.set('trust proxy', trustProxy);
    }

    const clientOrigin = configService.get<string>('CLIENT_ORIGIN', 'http://localhost:3001');
    const clientPublicUrl = configService.get<string>('CLIENT_PUBLIC_URL', clientOrigin);
    const csrfOriginCheckEnabled = configService.get<boolean>('CSRF_ORIGIN_CHECK_ENABLED', false);
    const port = configService.get<number>('PORT', 3000);

    app.use(createSecurityHeadersMiddleware(configService.get<string>('NODE_ENV', 'development')));
    app.use(createRequestIdMiddleware(loggerConfig));
    app.use(json({ limit: '8mb' }));
    app.use(urlencoded({ extended: true, limit: '8mb' }));
    app.use('/uploads', expressStatic(join(process.cwd(), 'uploads')));
    app.use(cookieParser());
    app.use(
        createCsrfOriginMiddleware({
            enabled: csrfOriginCheckEnabled,
            allowedOrigins: [clientOrigin, clientPublicUrl],
        }),
    );
    app.enableCors({
        origin: clientOrigin,
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', loggerConfig.requestIdHeader],
        exposedHeaders: [loggerConfig.requestIdHeader],
    });
    app.useGlobalPipes(
        new ValidationPipe({
            transform: true,
            whitelist: true,
            forbidNonWhitelisted: true,
        }),
    );
    setupSwagger(app);
    await app.listen(port);
}

bootstrap().catch((error: unknown) => {
    try {
        const appLogger = createStructuredBootstrapLoggerFromEnv();

        if (appLogger) {
            appLogger.error('Failed to start server', error, 'Bootstrap');
        } else {
            console.error('Failed to start server:', error);
        }
    } catch {
        console.error('Failed to start server:', error);
    }
    process.exit(1);
});
