import { Processor, WorkerHost } from '@nestjs/bullmq';
import { Injectable, Logger, OnApplicationBootstrap } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { getMailBootstrapConfig } from '@config/mail-bootstrap.config';
import { Job, UnrecoverableError } from 'bullmq';
import { MAIL_POISON_FAILURE_CODES, MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import type { MailPoisonFailureCode } from '@/module-mail/constants/mail.constants';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { SmtpMailService } from '@/module-mail/services/smtp-mail.service';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { SendMailJobPayload } from '@/module-mail/interfaces/mail-service.interface';

@Injectable()
@Processor(MAIL_QUEUE, { autorun: false })
export class MailProcessorService extends WorkerHost implements OnApplicationBootstrap {
    private readonly logger = new Logger(MailProcessorService.name);

    constructor(
        private readonly mailReadinessService: MailReadinessService,
        private readonly smtpMailService: SmtpMailService,
        private readonly mailEncryptionService: MailEncryptionService,
        private readonly mailOutboxRepository: MailOutboxRepository,
        private readonly configService: ConfigService,
    ) {
        super();
    }

    onApplicationBootstrap() {
        const config = getMailBootstrapConfig(this.configService);
        if (this.worker) {
            this.worker.opts.limiter = {
                max: config.queueLimiterMax,
                duration: config.queueLimiterDuration,
            };
            this.worker.concurrency = config.workerConcurrency;
            this.logger.log(
                `Configured MailProcessor worker: concurrency=${config.workerConcurrency}, limiter=${config.queueLimiterMax}/${config.queueLimiterDuration}ms`,
            );
            void this.worker.run().catch((error: unknown) => {
                const message = error instanceof Error ? error.message : String(error);
                this.logger.error(`MailProcessor worker stopped unexpectedly: ${message}`);
            });
        }
    }

    async process(job: Job<any, any, string>): Promise<any> {
        if (!job.id || !(await this.mailOutboxRepository.isQueuedForDelivery(job.id))) {
            throw new Error('Outbox record is not confirmed for delivery');
        }

        const transientReasons = [
            'too_many_delivery_failures',
            'smtp_verification_failed',
            'smtp_connection_failed',
            'smtp_auth_failed',
            'smtp_rate_limit',
            'smtp_send_failed',
        ];

        const health = await this.mailReadinessService.getHealth();
        if (health.status !== 'healthy') {
            const reason = health.lastError || 'Unknown configuration issue';

            // Circuit breaker/transient failures: pause the worker to defer processing
            if (transientReasons.includes(reason)) {
                this.logger.warn(
                    `Mail system is not healthy (${reason}). Pausing worker to defer processing.`,
                );
                if (this.worker && typeof this.worker.pause === 'function') {
                    const isAlreadyPaused =
                        typeof this.worker.isPaused === 'function' ? this.worker.isPaused() : false;
                    if (!isAlreadyPaused) {
                        await this.worker.pause();

                        // Schedule resume after 60 seconds
                        setTimeout(() => {
                            this.logger.log(
                                'Resuming MailProcessor worker after circuit breaker timeout.',
                            );
                            if (this.worker) {
                                const promise = this.worker.resume() as unknown as Promise<void>;
                                promise.catch((err: unknown) => {
                                    const errMsg = err instanceof Error ? err.message : String(err);
                                    this.logger.error(`Failed to resume worker: ${errMsg}`);
                                });
                            }
                        }, 60000).unref();
                    }
                }
            }
            throw new Error(`Mail system is not healthy: ${reason}`);
        }

        const payload = await this.resolvePayload(job);
        if (!payload) {
            return;
        }

        // 3. Send
        try {
            await this.smtpMailService.sendMailDirect({
                to: payload.to,
                subject: payload.subject,
                html: payload.html,
                text: payload.text,
                replyTo: payload.replyTo,
            });
            await this.mailReadinessService.setReadinessStatus('healthy');
        } catch (error: unknown) {
            const safeCode = this.getSafeErrorCode(error);
            await this.mailReadinessService.setReadinessStatus('unhealthy', safeCode, 120);

            // Check if this failure tripped the circuit breaker
            const postSendHealth = await this.mailReadinessService.getHealth();
            if (
                postSendHealth.status !== 'healthy' &&
                transientReasons.includes(postSendHealth.lastError || '')
            ) {
                this.logger.warn(
                    `SMTP send failed and tripped circuit breaker (${postSendHealth.lastError}). Pausing worker.`,
                );
                if (this.worker && typeof this.worker.pause === 'function') {
                    const isAlreadyPaused =
                        typeof this.worker.isPaused === 'function' ? this.worker.isPaused() : false;
                    if (!isAlreadyPaused) {
                        await this.worker.pause();

                        // Schedule resume after 60 seconds
                        setTimeout(() => {
                            this.logger.log(
                                'Resuming MailProcessor worker after circuit breaker timeout.',
                            );
                            if (this.worker) {
                                const promise = this.worker.resume() as unknown as Promise<void>;
                                promise.catch((err: unknown) => {
                                    const errMsg = err instanceof Error ? err.message : String(err);
                                    this.logger.error(`Failed to resume worker: ${errMsg}`);
                                });
                            }
                        }, 60000).unref();
                    }
                }
            }
            throw error;
        }

        // 4. Mark outbox record as sent
        if (job.id) {
            await this.mailOutboxRepository.markAsSent(job.id);
        }
    }

    private async resolvePayload(job: Job<any, any, string>): Promise<SendMailJobPayload | null> {
        const jobData = job.data as Record<string, unknown>;
        let payload: unknown;

        if (typeof jobData.payloadEncrypted === 'string' && jobData.payloadEncrypted.length > 0) {
            try {
                const decryptedStr = this.mailEncryptionService.decryptMailSecret(
                    jobData.payloadEncrypted,
                );
                payload = JSON.parse(decryptedStr);
            } catch {
                await this.markPoisonJob(job, MAIL_POISON_FAILURE_CODES.DECRYPTION_FAILURE);
                return null;
            }
        } else {
            await this.markPoisonJob(job, MAIL_POISON_FAILURE_CODES.PLAINTEXT_PAYLOAD_FORBIDDEN);
            return null;
        }

        if (!this.isValidMailJobPayload(payload)) {
            await this.markPoisonJob(job, MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA);
            return null;
        }

        return payload;
    }

    private async markPoisonJob(
        job: Job<any, any, string>,
        code: MailPoisonFailureCode,
    ): Promise<void> {
        this.logger.warn(`Rejecting poison mail job ${job.id ?? 'unknown'} with code ${code}`);
        if (job.id) {
            await this.mailOutboxRepository.markAsFailed(job.id, code);
        }
        throw new UnrecoverableError(code);
    }

    private isValidMailJobPayload(payload: unknown): payload is SendMailJobPayload {
        if (!payload || typeof payload !== 'object') {
            return false;
        }

        const candidate = payload as Partial<SendMailJobPayload>;
        const hasRecipient =
            typeof candidate.to === 'string'
                ? candidate.to.trim().length > 0
                : Array.isArray(candidate.to) &&
                  candidate.to.length > 0 &&
                  candidate.to.every(item => typeof item === 'string' && item.trim().length > 0);

        return (
            hasRecipient &&
            typeof candidate.subject === 'string' &&
            candidate.subject.trim().length > 0 &&
            typeof candidate.html === 'string' &&
            candidate.html.trim().length > 0 &&
            typeof candidate.text === 'string' &&
            candidate.text.trim().length > 0 &&
            (candidate.replyTo === undefined || typeof candidate.replyTo === 'string')
        );
    }

    private getSafeErrorCode(error: unknown): string {
        if (!error) return 'smtp_send_failed';
        const errObj = error as Record<string, unknown>;
        const code = typeof errObj.code === 'string' ? errObj.code.toUpperCase() : '';
        const message = typeof errObj.message === 'string' ? errObj.message.toUpperCase() : '';

        if (
            code === 'EAUTH' ||
            Number(errObj.responseCode) === 535 ||
            message.includes('535') ||
            message.includes('AUTHENTICATION') ||
            message.includes('CREDENTIALS')
        ) {
            return 'smtp_auth_failed';
        }

        const responseCode = Number(errObj.responseCode);
        if (
            responseCode === 421 ||
            message.includes('RATE LIMIT') ||
            message.includes('THROTTLE') ||
            message.includes('TOO MANY REQUESTS')
        ) {
            return 'smtp_rate_limit';
        }

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
            return 'smtp_connection_failed';
        }

        return 'smtp_send_failed';
    }
}
