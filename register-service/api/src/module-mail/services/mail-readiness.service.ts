import { forwardRef, Inject, Injectable, ServiceUnavailableException } from '@nestjs/common';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailHealth, MailHealthStatus } from '@/module-mail/interfaces/mail-service.interface';
import { MailSettingsSelect } from '@/module-mail/schemas';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';

@Injectable()
export class MailReadinessService {
    private localReadinessFallback: {
        status: string;
        error: string | null;
        timestamp: string;
    } | null = null;

    constructor(
        private readonly mailRedisService: MailRedisService,
        private readonly mailEncryptionService: MailEncryptionService,
        @Inject(forwardRef(() => MailSettingsService))
        private readonly mailSettingsService: MailSettingsService,
    ) {}

    /**
     * Evaluates the internal readiness of the mail system.
     * This is an internal-only check and must not be exposed as a public API.
     * It does NOT perform live SMTP verification.
     */
    async getHealth(): Promise<MailHealth> {
        const settings = await this.mailSettingsService.getCachedSettings();
        return await this.evaluateHealth(settings);
    }

    /**
     * Internal health evaluation logic.
     * Reusable to avoid redundant DB calls.
     */
    private async evaluateHealth(settings: MailSettingsSelect | null): Promise<MailHealth> {
        let redisStatus: {
            status: string;
            error: string | null;
            timestamp: string;
        } | null = null;
        let isRedisAvailable = true;

        try {
            redisStatus = await this.mailRedisService.getReadinessStatus();
        } catch {
            isRedisAvailable = false;
        }

        const effectiveStatus = redisStatus || this.localReadinessFallback;

        const failureCount = isRedisAvailable ? await this.mailRedisService.getFailureCount() : 0;
        const deliveryErrors = isRedisAvailable
            ? await this.mailRedisService.getDeliveryErrors()
            : [];
        const lastSuccessfulSendAt = isRedisAvailable
            ? await this.mailRedisService.getLastSuccessfulSendAt()
            : null;

        const baseHealth: MailHealth = {
            status: 'unhealthy',
            isDegraded: !isRedisAvailable,
            lastCheckedAt: new Date().toISOString(),
            lastSuccessfulSendAt,
            lastFailedSendAt: deliveryErrors[0]?.timestamp || null,
            lastError: deliveryErrors[0]?.message || null,
            aggregatedFailureCount: failureCount,
            deliveryErrors,
        };

        // 1. Explicit ephemeral status takes precedence
        if (effectiveStatus && effectiveStatus.status !== 'healthy') {
            return {
                ...baseHealth,
                status: effectiveStatus.status as MailHealthStatus,
                lastError: effectiveStatus.error || baseHealth.lastError,
            };
        }

        // 2. Database configuration checks
        if (!settings) {
            return { ...baseHealth, lastError: 'mail_settings_missing' };
        }

        if (!settings.enabled) {
            return { ...baseHealth, status: 'disabled' };
        }

        if (!settings.clientPublicUrl) {
            return { ...baseHealth, lastError: 'mail_settings_client_url_missing' };
        }

        if (!this.mailEncryptionService.isKeyAvailable()) {
            return { ...baseHealth, lastError: 'encryption_key_missing' };
        }

        // Check password decryptability if set
        if (settings.smtpPasswordEncrypted) {
            try {
                this.mailEncryptionService.decryptMailSecret(settings.smtpPasswordEncrypted);
            } catch {
                return { ...baseHealth, lastError: 'encryption_key_mismatch' };
            }
        }

        if (failureCount > 10) {
            return { ...baseHealth, lastError: 'too_many_delivery_failures' };
        }

        return {
            ...baseHealth,
            status: 'healthy',
        };
    }

    /**
     * Asserts that the mail system is ready for critical flows (password reset, email change).
     * It checks DB settings and internal readiness.
     */
    async assertMailReadyForCriticalFlow(): Promise<void> {
        const settings = await this.mailSettingsService.getCachedSettings();

        if (!settings || !settings.enabled) {
            throw new ServiceUnavailableException('mail_service_disabled_or_unconfigured');
        }

        const health = await this.evaluateHealth(settings);
        if (health.status !== 'healthy') {
            throw new ServiceUnavailableException(`mail_service_not_ready: ${health.lastError}`);
        }
    }

    /**
     * Sets a temporary readiness status (e.g. for circuit breaker).
     * This updates Redis (shared state) and falls back to local memory if Redis is down.
     */
    async setReadinessStatus(status: string, error?: string, ttlSeconds = 300): Promise<void> {
        try {
            await this.mailRedisService.setReadinessStatus(status, error, ttlSeconds);
        } catch {
            this.localReadinessFallback = {
                status,
                error: error || null,
                timestamp: new Date().toISOString(),
            };
            // Clear local fallback after TTL
            setTimeout(() => {
                this.localReadinessFallback = null;
            }, ttlSeconds * 1000).unref();
        }
    }
}
