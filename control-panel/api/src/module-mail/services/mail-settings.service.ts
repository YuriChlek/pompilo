import { Inject, Injectable, Logger, OnApplicationBootstrap } from '@nestjs/common';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailAuditEventRepository } from '@/module-mail/repository/mail-audit-event.repository';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailRedisInvalidationService } from '@/module-mail/services/mail-redis-invalidation.service';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MAIL_TEMPLATE_PREVIEW_REGISTRY } from '@/module-mail/constants/mail-template-preview.constants';
import * as React from 'react';
import { MailSetupState } from '@/module-mail/types/mail-settings.types';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import { UpdateMailSettingsDto } from '@/module-mail/dto/admin-mail.dto';
import { MailSettingsInsert, MailSettingsSelect } from '@/module-mail/schemas';
import type { MailAuditPayload } from '@/module-mail/interfaces/mail-audit-event.interfaces';
import type { MailService } from '@/module-mail/interfaces/mail-service.interface';

import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';

@Injectable()
export class MailSettingsService implements OnApplicationBootstrap {
    private readonly logger = new Logger(MailSettingsService.name);
    private cachedSettings: MailSettingsSelect | null = null;
    private cacheTimestamp: number = 0;
    private readonly CACHE_TTL_MS = 60 * 1000; // 1 minute TTL fallback

    constructor(
        private readonly transactionRepository: TransactionRepository,
        private readonly mailSettingsRepository: MailSettingsRepository,
        private readonly mailAuditEventRepository: MailAuditEventRepository,
        private readonly mailEncryptionService: MailEncryptionService,
        private readonly mailRedisInvalidationService: MailRedisInvalidationService,
        @Inject(MAIL_SERVICE) private readonly mailService: MailService,
        private readonly mailRenderService: MailRenderService,
    ) {}

    onApplicationBootstrap() {
        this.mailRedisInvalidationService.onInvalidate(() => {
            this.logger.log('Mail settings cache invalidated via Pub/Sub.');
            this.clearCache();
        });
    }

    private clearCache() {
        this.cachedSettings = null;
        this.cacheTimestamp = 0;
    }

    async getCachedSettings(
        transaction?: RepositoryTransaction,
    ): Promise<MailSettingsSelect | null> {
        if (transaction) {
            return await this.mailSettingsRepository.findSingleton(transaction);
        }

        const now = Date.now();
        if (this.cachedSettings && now - this.cacheTimestamp < this.CACHE_TTL_MS) {
            return this.cachedSettings;
        }

        const settings = await this.mailSettingsRepository.findSingleton();
        this.cachedSettings = settings;
        this.cacheTimestamp = now;
        return settings;
    }

    async getSetupState(): Promise<MailSetupState> {
        const settings = await this.getCachedSettings();

        if (!settings) {
            return 'unconfigured';
        }

        return 'configured';
    }

    async getSettings(): Promise<unknown> {
        return this.getMaskedSettings();
    }

    private async getMaskedSettings(transaction?: RepositoryTransaction): Promise<unknown> {
        const settings = await this.getCachedSettings(transaction);
        if (!settings) {
            return null;
        }
        return this.mailEncryptionService.maskMailSecretPresence(settings) as unknown;
    }

    async updateSettings(dto: UpdateMailSettingsDto, userId: string): Promise<unknown> {
        return this.transactionRepository.run(async transaction => {
            let existingSettings = await this.mailSettingsRepository.findSingleton(transaction);

            // Prepare base update data
            const updateData: Partial<MailSettingsInsert> = {
                ...dto,
                updatedByUserId: userId,
            };

            if (dto.smtpPassword) {
                updateData.smtpPasswordEncrypted = this.mailEncryptionService.encryptMailSecret(
                    dto.smtpPassword,
                );
            }

            if (!existingSettings) {
                const created = await this.mailSettingsRepository.create(
                    updateData as MailSettingsInsert,
                    transaction,
                );
                // If conflict occurred and created is null, re-read to get the actual row
                if (!created) {
                    existingSettings = await this.mailSettingsRepository.findSingleton(transaction);
                    if (existingSettings) {
                        await this.mailSettingsRepository.update(
                            existingSettings.id,
                            updateData,
                            transaction,
                        );
                    }
                }
            } else {
                await this.mailSettingsRepository.update(
                    existingSettings.id,
                    updateData,
                    transaction,
                );
            }

            this.clearCache();
            await this.mailRedisInvalidationService.invalidate();

            await this.mailAuditEventRepository.create(
                {
                    action:
                        dto.enabled === false ? 'mail_settings_disabled' : 'mail_settings_updated',
                    adminUserId: userId,
                    payload: this.buildSettingsAuditPayload(dto),
                },
                transaction,
            );

            return this.getMaskedSettings(transaction);
        });
    }

    async sendTestEmail(to: string, templateId: string, adminUserId: string, adminEmail: string) {
        const recipientMatchesAdmin = this.isSameEmailAddress(to, adminEmail);
        const registryItem = MAIL_TEMPLATE_PREVIEW_REGISTRY[templateId];
        if (!registryItem) {
            throw new Error(`Mail template with ID "${templateId}" not found`);
        }

        const component = React.createElement(registryItem.component, registryItem.demoProps);
        const html = await this.mailRenderService.renderHtml(component);
        const text = await this.mailRenderService.renderText(component);

        const deliveryRequest = await this.mailService.createDeliveryRequest({
            to,
            subject: registryItem.subject,
            html,
            text,
        });

        if (!deliveryRequest) {
            await this.createAuditEvent(adminUserId, 'mail_test_email_requested', {
                errorCode: 'delivery_request_not_created',
                recipientMatchesAdmin,
            });
            return null;
        }
        await this.createAuditEvent(adminUserId, 'mail_test_email_requested', {
            outboxId: deliveryRequest.outboxId,
            recipientMatchesAdmin,
        });

        return deliveryRequest;
    }

    private isSameEmailAddress(firstEmail: string, secondEmail: string): boolean {
        return firstEmail.trim().toLowerCase() === secondEmail.trim().toLowerCase();
    }

    private buildSettingsAuditPayload(dto: UpdateMailSettingsDto): MailAuditPayload {
        const changedFields = Object.keys(dto)
            .filter(field => field !== 'smtpPassword')
            .sort();
        const payload: MailAuditPayload = {
            changedFields,
        };

        if (Object.prototype.hasOwnProperty.call(dto, 'smtpPassword')) {
            payload.hasSmtpPasswordChange = Boolean(dto.smtpPassword);
        }

        if (dto.enabled !== undefined) {
            payload.enabled = dto.enabled;
        }

        return payload;
    }

    private async createAuditEvent(
        adminUserId: string,
        action: Parameters<MailAuditEventRepository['create']>[0]['action'],
        payload: MailAuditPayload,
    ): Promise<void> {
        await this.mailAuditEventRepository.create({
            action,
            adminUserId,
            payload,
        });
    }
}
