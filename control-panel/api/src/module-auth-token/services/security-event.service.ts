import { Injectable, Logger } from '@nestjs/common';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import type { SecurityEventWriteInput } from '@/module-auth-token/interfaces/security-event.interfaces';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';
import { SecurityEventSelect } from '@/module-auth-token/schemas/security-events.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';

@Injectable()
export class SecurityEventService {
    private readonly logger = new Logger(SecurityEventService.name);

    public constructor(private readonly securityEventRepository: SecurityEventRepository) {}

    async recordSecurityEvent(
        data: SecurityEventWriteInput,
        transaction?: RepositoryTransaction,
        failurePolicy: 'critical' | 'best-effort' = 'critical',
    ): Promise<SecurityEventSelect | null> {
        try {
            return await this.securityEventRepository.save(data, transaction);
        } catch (error) {
            this.logger.error(
                `Security event recording failed [type=${data.eventType}, policy=${failurePolicy}]: ${
                    error instanceof Error ? error.message : String(error)
                }`,
                error instanceof Error ? error.stack : undefined,
            );
            if (failurePolicy === 'critical') {
                throw error;
            }
            return null;
        }
    }

    async recordLoginSuccess(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.LOGIN_SUCCESS }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_SUCCESS,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordRegistrationSuccess(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.REGISTRATION_SUCCESS }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.REGISTRATION_SUCCESS,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginFailed(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.LOGIN_FAILED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_FAILED,
            } as SecurityEventWriteInput,
            transaction,
            'best-effort',
        );
    }

    async recordRefreshFailed(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.REFRESH_FAILED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.REFRESH_FAILED,
            } as SecurityEventWriteInput,
            transaction,
            'best-effort',
        );
    }

    async recordLogout(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.LOGOUT }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGOUT,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordSessionRevoked(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.SESSION_REVOKED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.SESSION_REVOKED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordRevokeOtherSessions(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.REVOKE_OTHER_SESSIONS }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.REVOKE_OTHER_SESSIONS,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordRevokeAllSessions(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.REVOKE_ALL_SESSIONS }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.REVOKE_ALL_SESSIONS,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordPasswordChanged(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.PASSWORD_CHANGED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.PASSWORD_CHANGED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordPasswordResetCompleted(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.PASSWORD_RESET_COMPLETED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordEmailChangeRequested(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.EMAIL_CHANGE_REQUESTED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.EMAIL_CHANGE_REQUESTED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordEmailVerified(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.EMAIL_VERIFIED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.EMAIL_VERIFIED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordSuspiciousDevice(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.SUSPICIOUS_DEVICE }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.SUSPICIOUS_DEVICE,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalRequired(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_REQUIRED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_REQUIRED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalPassed(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_PASSED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_PASSED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalFailed(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_FAILED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_FAILED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalExpired(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_EXPIRED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_EXPIRED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalResent(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_RESENT }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_RESENT,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordLoginApprovalResendFailed(
        data: Omit<
            Extract<
                SecurityEventWriteInput,
                { eventType: SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED }
            >,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED,
            } as SecurityEventWriteInput,
            transaction,
            'best-effort',
        );
    }

    async recordReauthPassed(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.RE_AUTH_PASSED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.RE_AUTH_PASSED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async recordReauthFailed(
        data: Omit<
            Extract<SecurityEventWriteInput, { eventType: SecurityEventType.RE_AUTH_FAILED }>,
            'eventType'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<SecurityEventSelect | null> {
        return await this.recordSecurityEvent(
            {
                ...data,
                eventType: SecurityEventType.RE_AUTH_FAILED,
            } as SecurityEventWriteInput,
            transaction,
            'critical',
        );
    }

    async deleteExpiredEvents(retentionPeriodMs = 0): Promise<number> {
        return await this.securityEventRepository.deleteExpiredEvents(retentionPeriodMs);
    }
}
