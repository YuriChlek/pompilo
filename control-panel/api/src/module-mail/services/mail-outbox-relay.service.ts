import { Injectable, Logger, OnApplicationBootstrap, OnApplicationShutdown } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { InjectQueue } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { Client } from 'pg';
import { randomUUID } from 'crypto';
import { ConfigService } from '@nestjs/config';
import { getMailBootstrapConfig } from '@config/mail-bootstrap.config';
import { MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { getDrizzleDbConfig } from '@/module-drizzle/db-config/drizzle-db.config';

@Injectable()
export class MailOutboxRelayService implements OnApplicationBootstrap, OnApplicationShutdown {
    private readonly logger = new Logger(MailOutboxRelayService.name);
    private readonly BATCH_SIZE = 50;
    private readonly instanceId = randomUUID(); // Uniquely identifies this relay instance
    private isRunning = false;
    private isRecoveryRunning = false;
    private hasPendingNotification = false;
    private readonly STALE_TIMEOUT_MINUTES = 5;

    private pgListenerClient: Client | null = null;
    private isListening = false;

    constructor(
        private readonly mailOutboxRepository: MailOutboxRepository,
        private readonly mailReadinessService: MailReadinessService,
        private readonly configService: ConfigService,
        @InjectQueue(MAIL_QUEUE) private readonly mailQueue: Queue,
    ) {}

    async onApplicationBootstrap() {
        // Run stale lock recovery first
        await this.handleStaleLocks();
        // Start PostgreSQL LISTEN listener
        await this.startPgListener();
    }

    async onApplicationShutdown() {
        await this.stopPgListener();
    }

    private async startPgListener() {
        if (this.isListening && this.pgListenerClient) {
            return;
        }
        this.isListening = true;

        const dbConfig = getDrizzleDbConfig(this.configService);

        const connectAndListen = async (attempt = 1) => {
            if (!this.isListening) {
                return;
            }

            try {
                this.pgListenerClient = new Client(dbConfig);
                await this.pgListenerClient.connect();

                this.pgListenerClient.on('error', err => {
                    this.logger.error(`PostgreSQL listener client error: ${err.message}`);
                    this.reconnectPgListener(attempt + 1).catch((reconnectErr: Error) => {
                        this.logger.error(`Error during reconnect: ${reconnectErr.message}`);
                    });
                });

                this.pgListenerClient.on('notification', msg => {
                    if (msg.channel === 'mail_outbox_inserted') {
                        this.triggerRelay(true).catch((relayErr: Error) => {
                            this.logger.error(`Error during triggerRelay: ${relayErr.message}`);
                        });
                    }
                });

                await this.pgListenerClient.query('LISTEN mail_outbox_inserted');
                this.logger.log('PostgreSQL LISTEN trigger registered for mail_outbox_inserted.');
            } catch (error: unknown) {
                const message = error instanceof Error ? error.message : String(error);
                this.logger.error(
                    `Failed to connect PostgreSQL listener (attempt ${attempt}): ${message}`,
                );
                await this.reconnectPgListener(attempt + 1);
            }
        };

        await connectAndListen();
    }

    private async reconnectPgListener(nextAttempt: number) {
        if (this.pgListenerClient) {
            try {
                await this.pgListenerClient.end();
            } catch {
                // Ignore client closing errors
            }
            this.pgListenerClient = null;
        }

        if (!this.isListening) {
            return;
        }

        const delay = Math.min(1000 * Math.pow(2, nextAttempt), 30000);
        this.logger.warn(`Reconnecting PostgreSQL listener in ${delay}ms...`);
        setTimeout(() => {
            this.startPgListener().catch((startErr: Error) => {
                this.logger.error(`Error starting listener: ${startErr.message}`);
            });
        }, delay).unref();
    }

    private async stopPgListener() {
        this.isListening = false;
        if (this.pgListenerClient) {
            try {
                await this.pgListenerClient.end();
            } catch {
                // Ignore client closing errors
            }
            this.pgListenerClient = null;
        }
    }

    private async triggerRelay(isEventDriven = false) {
        if (this.isRunning) {
            if (isEventDriven) {
                this.hasPendingNotification = true;
            }
            return;
        }
        this.isRunning = true;

        try {
            do {
                this.hasPendingNotification = false;
                await this.relayOutbox();
            } while (this.hasPendingNotification);
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Error during outbox relay: ${message}`);
            await this.mailReadinessService.setReadinessStatus(
                'unhealthy',
                `Relay loop error: ${message}`,
                600,
            );
        } finally {
            this.isRunning = false;
        }
    }

    @Cron(CronExpression.EVERY_MINUTE)
    async handleStaleLocks() {
        if (this.isRecoveryRunning) {
            return;
        }
        this.isRecoveryRunning = true;

        try {
            const config = getMailBootstrapConfig(this.configService);
            const reclaimed = await this.mailOutboxRepository.releaseStaleLocks(
                this.STALE_TIMEOUT_MINUTES,
                config.retryAttempts,
            );
            if (reclaimed > 0) {
                this.logger.warn(`Reclaimed ${reclaimed} stale outbox records.`);
            }
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Error during stale locks recovery: ${message}`);
        } finally {
            this.isRecoveryRunning = false;
        }
    }

    @Cron(CronExpression.EVERY_5_MINUTES)
    async handleCron() {
        await this.triggerRelay();
    }

    private async relayOutbox() {
        let iterations = 0;
        const MAX_ITERATIONS = 10;
        let processedInIteration = 0;

        do {
            processedInIteration = await this.relayBatch();
            iterations++;

            if (processedInIteration > 0 && iterations < MAX_ITERATIONS) {
                // Cooperative yielding: release the Event Loop
                await new Promise(resolve => setImmediate(resolve));
            }
        } while (processedInIteration === this.BATCH_SIZE && iterations < MAX_ITERATIONS);
    }

    private async relayBatch(): Promise<number> {
        const records = await this.mailOutboxRepository.claimRecords(
            this.BATCH_SIZE,
            this.instanceId,
        );

        if (records.length === 0) {
            return 0;
        }

        const config = getMailBootstrapConfig(this.configService);

        for (const record of records) {
            try {
                // If it already has a queuedJobId or is queued (safety check), skip it
                if (record.queuedJobId || record.status === 'queued') {
                    this.logger.warn(`Record ${record.id} is already queued, skipping.`);
                    continue;
                }

                let jobPayload: Record<string, unknown> = {};

                if (record.payloadJson) {
                    jobPayload = record.payloadJson as Record<string, unknown>;
                } else if (record.payloadEncrypted) {
                    // Processor will decrypt it, we just pass it along
                    jobPayload = { payloadEncrypted: record.payloadEncrypted };
                }

                // Attach idempotencyKey to the payload so processor can use it if needed
                jobPayload.idempotencyKey = record.idempotencyKey;
                // Add to BullMQ with explicit retention, backoff and stable jobId
                const job = await this.mailQueue.add('send-mail', jobPayload, {
                    jobId: record.idempotencyKey,
                    attempts: config.retryAttempts,
                    backoff: {
                        type: (config.retryBackoffType || 'exponential') as 'exponential' | 'fixed',
                        delay: config.retryBackoffDelay ?? 2000,
                        jitter: config.retryBackoffJitter ?? 0.2,
                    },
                    removeOnComplete: {
                        age: 3600,
                        count: 1000,
                    },
                    removeOnFail: {
                        age: 3600, // 1 hour max
                        count: 50,
                    },
                });

                if (!job?.id) {
                    throw new Error('BullMQ returned a job without an id');
                }

                try {
                    const markedAsQueued = await this.mailOutboxRepository.markAsQueued(
                        record.id,
                        job.id,
                        this.instanceId,
                    );
                    if (!markedAsQueued) {
                        throw new Error(
                            'Outbox claim ownership was lost before queue confirmation',
                        );
                    }
                } catch (error: unknown) {
                    try {
                        await job.remove();
                    } catch (removeError: unknown) {
                        const removeMessage =
                            removeError instanceof Error
                                ? removeError.message
                                : String(removeError);
                        this.logger.error(
                            `Failed to compensate BullMQ job ${job.id} after outbox update failure: ${removeMessage}`,
                        );
                        throw error;
                    }
                    throw error;
                }
            } catch (error: unknown) {
                const message = error instanceof Error ? error.message : String(error);
                this.logger.error(`Failed to publish outbox record ${record.id}: ${message}`);

                // Phase 3: Signal degraded state on publish failure
                await this.mailReadinessService.setReadinessStatus(
                    'unhealthy',
                    `Relay publish failed: ${message}`,
                    600,
                );

                try {
                    await this.mailOutboxRepository.markPublishFailedForRetry(
                        record.id,
                        message,
                        config.retryAttempts,
                    );
                } catch (releaseError: unknown) {
                    const releaseMessage =
                        releaseError instanceof Error ? releaseError.message : String(releaseError);
                    this.logger.error(
                        `Failed to release outbox record ${record.id} after publish failure: ${releaseMessage}`,
                    );
                }
            }
        }

        return records.length;
    }
}
