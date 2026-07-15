export type MailHealthStatus = 'healthy' | 'unhealthy' | 'disabled';

export interface MailHealth {
    status: MailHealthStatus;
    isDegraded: boolean;
    lastCheckedAt: string | null;
    lastSuccessfulSendAt: string | null;
    lastFailedSendAt: string | null;
    lastError: string | null;
    aggregatedFailureCount: number;
    deliveryErrors: Array<{ code: string; message: string; timestamp: string }>;
}

export interface SendMailPayload {
    to: string | string[];
    subject: string;
    html: string;
    text: string;
    replyTo?: string;
    attachments?: Array<{
        filename: string;
        content: string | Buffer;
        contentType?: string;
    }>;
}

export interface MailDeliveryRequest {
    outboxId: string;
    idempotencyKey: string;
    status: 'accepted';
}

export interface SendMailJobPayload {
    idempotencyKey: string;
    to: string | string[];
    subject: string;
    html: string;
    text: string;
    replyTo?: string;
    templateName?: string;
    metadata?: Record<string, string | number | boolean | null>;
}

export interface MailService {
    createDeliveryRequest(
        payload: SendMailPayload,
        transaction?: RepositoryTransaction,
    ): Promise<MailDeliveryRequest>;
    verifyTransport(settings: any): Promise<void>;
}
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
