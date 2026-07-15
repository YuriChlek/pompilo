import { Injectable, Logger } from '@nestjs/common';
import * as nodemailer from 'nodemailer';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import {
    MailDeliveryRequest,
    MailService,
    SendMailPayload,
} from '@/module-mail/interfaces/mail-service.interface';
import { MailRedisInvalidationService } from './mail-redis-invalidation.service';
import { MailReadinessService } from './mail-readiness.service';
import { MailSettingsSelect } from '@/module-mail/schemas';

import { randomUUID } from 'crypto';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { redactMailRecipients } from '@/module-mail/utils/mail-log-redaction.util';

@Injectable()
export class SmtpMailService implements MailService {
    private readonly logger = new Logger(SmtpMailService.name);
    private transport: nodemailer.Transporter | null = null;
    private configHash: string | null = null;
    private cachedSettings: MailSettingsSelect | null = null;
    private cacheTimestamp: number = 0;
    private readonly CACHE_TTL_MS = 60 * 1000; // 1 minute TTL fallback

    constructor(
        private readonly mailSettingsRepository: MailSettingsRepository,
        private readonly mailOutboxRepository: MailOutboxRepository,
        private readonly mailEncryptionService: MailEncryptionService,
        private readonly mailRedisService: MailRedisService,
        private readonly invalidationService: MailRedisInvalidationService,
        private readonly mailReadinessService: MailReadinessService,
    ) {
        this.invalidationService.onInvalidate(() => this.resetTransport());
    }

    private resetTransport() {
        this.logger.log('Mail config invalidated, resetting transport...');
        this.transport = null;
        this.configHash = null;
        this.cachedSettings = null;
        this.cacheTimestamp = 0;
    }

    private async getCachedSettings(): Promise<MailSettingsSelect | null> {
        const now = Date.now();
        if (this.cachedSettings && now - this.cacheTimestamp < this.CACHE_TTL_MS) {
            return this.cachedSettings;
        }

        const settings = await this.mailSettingsRepository.findSingleton();
        this.cachedSettings = settings;
        this.cacheTimestamp = now;
        return settings;
    }

    private async getTransport(): Promise<nodemailer.Transporter> {
        const settings = await this.getCachedSettings();
        if (!settings || !settings.enabled) {
            throw new Error('mail_service_disabled_or_unconfigured');
        }

        const currentHash = this.calculateConfigHash(settings);
        if (this.transport && this.configHash === currentHash) {
            return this.transport;
        }

        let password = '';
        if (settings.smtpPasswordEncrypted) {
            try {
                password = this.mailEncryptionService.decryptMailSecret(
                    settings.smtpPasswordEncrypted,
                );
            } catch (error: unknown) {
                const message = error instanceof Error ? error.message : String(error);
                this.logger.error(`Failed to decrypt SMTP password: ${message}`);
                throw new Error('mail_config_decryption_failed');
            }
        }

        this.logger.log(
            `Creating new SMTP transport for ${settings.smtpHost}:${settings.smtpPort}`,
        );

        this.transport = nodemailer.createTransport({
            host: settings.smtpHost,
            port: settings.smtpPort,
            secure: settings.smtpSecure,
            auth: settings.smtpUser
                ? {
                      user: settings.smtpUser,
                      pass: password,
                  }
                : undefined,
        });

        this.configHash = currentHash;

        try {
            await this.transport.verify();
            this.logger.log('SMTP transport verified successfully.');
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`SMTP verification failed: ${message}`);

            // Trigger circuit breaker for verification failures
            await this.mailReadinessService.setReadinessStatus(
                'unhealthy',
                'smtp_verification_failed',
                600, // 10 minutes
            );

            this.transport = null;
            this.configHash = null;
            throw new Error('smtp_verification_failed');
        }

        return this.transport;
    }

    private calculateConfigHash(settings: MailSettingsSelect): string {
        // Simple hash of settings that affect the transport
        return [
            settings.smtpHost,
            settings.smtpPort,
            settings.smtpSecure,
            settings.smtpUser,
            settings.smtpPasswordEncrypted,
        ].join('|');
    }

    async createDeliveryRequest(
        payload: SendMailPayload,
        transaction?: RepositoryTransaction,
    ): Promise<MailDeliveryRequest> {
        const idempotencyKey = randomUUID();
        const payloadJsonString = JSON.stringify(payload);
        const encryptedPayload = this.mailEncryptionService.encryptMailSecret(payloadJsonString);

        const outbox = await this.mailOutboxRepository.create(
            {
                idempotencyKey,
                payloadEncrypted: encryptedPayload,
                priority: 10, // Higher priority for critical flows
            },
            transaction,
        );

        return {
            outboxId: outbox.id,
            idempotencyKey: outbox.idempotencyKey,
            status: 'accepted',
        };
    }

    private isConnectionOrAuthError(error: unknown): boolean {
        if (!error) return false;
        const errObj = error as Record<string, unknown>;
        const code = typeof errObj.code === 'string' ? errObj.code.toUpperCase() : '';
        const message = typeof errObj.message === 'string' ? errObj.message.toUpperCase() : '';

        const connectionCodes = [
            'ECONNREFUSED',
            'ETIMEDOUT',
            'ECONNRESET',
            'EADDRNOTAVAIL',
            'ENOTFOUND',
            'EAI_AGAIN',
        ];

        const authCodes = ['EAUTH'];

        if (connectionCodes.includes(code) || authCodes.includes(code)) {
            return true;
        }

        const responseCode = Number(errObj.responseCode);
        if (responseCode === 535) {
            return true;
        }

        if (
            message.includes('535') ||
            message.includes('AUTHENTICATION') ||
            message.includes('CREDENTIALS') ||
            message.includes('CONNECT') ||
            message.includes('TIMEOUT') ||
            message.includes('REFUSED') ||
            message.includes('DNS')
        ) {
            return true;
        }

        return false;
    }

    private classifyError(error: unknown): 'network' | 'auth' | 'rate-limit' | 'rejection' {
        if (!error) return 'rejection';
        const errObj = error as Record<string, unknown>;
        const code = typeof errObj.code === 'string' ? errObj.code.toUpperCase() : '';
        const message = typeof errObj.message === 'string' ? errObj.message.toUpperCase() : '';

        // 1. Auth check
        if (
            code === 'EAUTH' ||
            Number(errObj.responseCode) === 535 ||
            message.includes('535') ||
            message.includes('AUTHENTICATION') ||
            message.includes('CREDENTIALS')
        ) {
            return 'auth';
        }

        // 2. Rate limit check
        const responseCode = Number(errObj.responseCode);
        if (
            responseCode === 421 ||
            message.includes('RATE LIMIT') ||
            message.includes('THROTTLE') ||
            message.includes('TOO MANY REQUESTS')
        ) {
            return 'rate-limit';
        }

        // 3. Network check
        const connectionCodes = [
            'ECONNREFUSED',
            'ETIMEDOUT',
            'ECONNRESET',
            'EADDRNOTAVAIL',
            'ENOTFOUND',
            'EAI_AGAIN',
        ];
        if (
            connectionCodes.includes(code) ||
            message.includes('CONNECT') ||
            message.includes('TIMEOUT') ||
            message.includes('REFUSED') ||
            message.includes('DNS')
        ) {
            return 'network';
        }

        // 4. Default is rejection
        return 'rejection';
    }

    async sendMailDirect(payload: SendMailPayload): Promise<void> {
        const transport = await this.getTransport();
        const settings = await this.getCachedSettings();

        if (!settings) throw new Error('mail_settings_missing');

        const from = `"${settings.fromName}" <${settings.fromAddress}>`;
        const redactedRecipients = redactMailRecipients(payload.to);

        try {
            await transport.sendMail({
                from,
                to: payload.to,
                subject: payload.subject,
                text: payload.text,
                html: payload.html,
                replyTo: payload.replyTo || settings.replyTo || undefined,
                attachments: payload.attachments,
            });
            await this.mailRedisService.logSuccessfulDelivery();
            await this.mailRedisService.incrementSuccessCount();
            this.logger.log(`Email sent successfully to ${redactedRecipients}`);
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Failed to send email to ${redactedRecipients}: ${message}`);

            const category = this.classifyError(error);
            await this.mailRedisService.incrementErrorCount(category);

            if (this.isConnectionOrAuthError(error)) {
                const failureCount = await this.mailRedisService.incrementFailureCount();
                await this.mailRedisService.logDeliveryError('smtp_send_failed', message);

                // Circuit breaker: if we have a burst of connection/auth failures, trip it
                if (failureCount >= 5) {
                    await this.mailReadinessService.setReadinessStatus(
                        'unhealthy',
                        'too_many_delivery_failures',
                        120, // 2 minutes (as per Phase 30: "1-2 хвилини")
                    );
                }
            } else {
                await this.mailRedisService.logDeliveryError('smtp_send_error', message);
            }

            throw error;
        }
    }

    async verifyTransport(settings: {
        smtpHost: string;
        smtpPort: number;
        smtpSecure: boolean;
        smtpUser?: string | null;
        smtpPassword?: string | null;
        smtpPasswordEncrypted?: string | null;
    }): Promise<void> {
        let password = settings.smtpPassword || '';
        if (!password && settings.smtpPasswordEncrypted) {
            try {
                password = this.mailEncryptionService.decryptMailSecret(
                    settings.smtpPasswordEncrypted,
                );
            } catch {
                throw new Error('mail_config_decryption_failed');
            }
        }

        const tempTransport = nodemailer.createTransport({
            host: settings.smtpHost,
            port: settings.smtpPort,
            secure: settings.smtpSecure,
            auth: settings.smtpUser
                ? {
                      user: settings.smtpUser,
                      pass: password,
                  }
                : undefined,
            // Phase 11: Bounded timeout for verification
            connectionTimeout: 10000, // 10 seconds
            greetingTimeout: 10000,
            socketTimeout: 10000,
        });

        try {
            await tempTransport.verify();
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Manual SMTP verification failed: ${message}`);
            throw new Error('smtp_verification_failed');
        } finally {
            tempTransport.close();
        }
    }
}
