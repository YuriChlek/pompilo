import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { SecurityEventSelect } from '@/module-auth-token/schemas/security-events.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import type { SecurityEventWriteInput } from '@/module-auth-token/interfaces/security-event.interfaces';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

const buildMockEvent = (overrides: Partial<SecurityEventSelect> = {}): SecurityEventSelect => ({
    id: overrides.id ?? 'event-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    sessionId: overrides.sessionId ?? 'session-uuid',
    knownDeviceId: overrides.knownDeviceId ?? 'device-uuid',
    eventType: overrides.eventType ?? SecurityEventType.LOGIN_SUCCESS,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
    ipAddress: overrides.ipAddress ?? null,
    country: overrides.country ?? null,
    region: overrides.region ?? null,
    city: overrides.city ?? null,
    userAgent: overrides.userAgent ?? null,
    metadata: overrides.metadata ?? null,
    createdAt: overrides.createdAt ?? new Date(),
});

describe('SecurityEventService', () => {
    let service: SecurityEventService;
    let repository: {
        save: jest.MockedFunction<SecurityEventRepository['save']>;
    };

    beforeEach(() => {
        repository = {
            save: jest.fn(),
        };

        service = new SecurityEventService(repository as unknown as SecurityEventRepository);
    });

    describe('recordSecurityEvent', () => {
        it('should call save on repository with input data and transaction', async () => {
            const input: SecurityEventWriteInput = {
                userId: 'user-uuid',
                realm: 'customer',
                sessionId: 'session-uuid',
                knownDeviceId: 'device-uuid',
                eventType: SecurityEventType.LOGIN_SUCCESS,
                metadata: {},
            };
            const event = buildMockEvent(input);
            repository.save.mockResolvedValue(event);

            const tx = {} as RepositoryTransaction;
            const result = await service.recordSecurityEvent(input, tx);

            expect(repository.save).toHaveBeenCalledWith(input, tx);
            expect(result).toEqual(event);
        });

        it('should propagate repository errors and log them under default critical policy', async () => {
            const input: SecurityEventWriteInput = {
                userId: 'user-uuid',
                realm: 'customer',
                sessionId: 'session-uuid',
                knownDeviceId: 'device-uuid',
                eventType: SecurityEventType.LOGIN_SUCCESS,
                metadata: {},
            };
            repository.save.mockRejectedValue(new Error('DB Save Failed'));

            await expect(service.recordSecurityEvent(input)).rejects.toThrow('DB Save Failed');
            expect(repository.save).toHaveBeenCalledWith(input, undefined);
        });

        it('should catch error, log it, and return null under best-effort policy', async () => {
            const input: SecurityEventWriteInput = {
                realm: 'customer',
                ipAddress: '1.2.3.4',
                eventType: SecurityEventType.LOGIN_FAILED,
                metadata: { failureReason: 'wrong_password' },
            };
            repository.save.mockRejectedValue(new Error('DB Outage'));

            const result = await service.recordSecurityEvent(input, undefined, 'best-effort');

            expect(result).toBeNull();
            expect(repository.save).toHaveBeenCalledWith(input, undefined);
        });

        it('should return null and not throw in recordLoginFailed if database save fails (best-effort wrapper check)', async () => {
            const data = {
                realm: 'customer' as const,
                ipAddress: '1.2.3.4',
                metadata: { failureReason: 'wrong_password' },
            };
            repository.save.mockRejectedValue(new Error('DB Outage'));

            const result = await service.recordLoginFailed(data);

            expect(result).toBeNull();
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_FAILED },
                undefined,
            );
        });
    });

    describe('typed helpers', () => {
        it('should correctly wrap recordLoginSuccess', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                knownDeviceId: 'k',
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.LOGIN_SUCCESS });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginSuccess(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_SUCCESS },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordRegistrationSuccess', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                knownDeviceId: 'k',
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.REGISTRATION_SUCCESS,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordRegistrationSuccess(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.REGISTRATION_SUCCESS },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginFailed', async () => {
            const data = {
                realm: 'customer' as const,
                ipAddress: 'ip',
                metadata: { failureReason: 'f' },
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.LOGIN_FAILED });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginFailed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_FAILED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordRefreshFailed', async () => {
            const data = {
                realm: 'customer' as const,
                ipAddress: 'ip',
                metadata: { failureReason: 'f' },
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.REFRESH_FAILED });
            repository.save.mockResolvedValue(event);
            const result = await service.recordRefreshFailed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.REFRESH_FAILED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLogout', async () => {
            const data = { userId: 'u', realm: 'customer' as const, sessionId: 's' };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.LOGOUT });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLogout(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGOUT },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordSessionRevoked', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { revocationReason: 'r' },
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.SESSION_REVOKED });
            repository.save.mockResolvedValue(event);
            const result = await service.recordSessionRevoked(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.SESSION_REVOKED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordRevokeOtherSessions', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { revokedSessionCount: 1 },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.REVOKE_OTHER_SESSIONS,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordRevokeOtherSessions(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.REVOKE_OTHER_SESSIONS },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordRevokeAllSessions', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                metadata: { revokedSessionCount: 1, revocationReason: 'r' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.REVOKE_ALL_SESSIONS,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordRevokeAllSessions(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.REVOKE_ALL_SESSIONS },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordPasswordChanged', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { revokedSessionCount: 1 },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.PASSWORD_CHANGED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordPasswordChanged(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.PASSWORD_CHANGED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordPasswordResetCompleted', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                metadata: { revokedSessionCount: 1 },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordPasswordResetCompleted(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.PASSWORD_RESET_COMPLETED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordEmailChangeRequested', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { newEmailHash: 'h' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.EMAIL_CHANGE_REQUESTED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordEmailChangeRequested(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.EMAIL_CHANGE_REQUESTED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordSuspiciousDevice', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                ipAddress: 'ip',
                metadata: { deviceId: 'd', riskSignals: {} },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.SUSPICIOUS_DEVICE,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordSuspiciousDevice(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.SUSPICIOUS_DEVICE },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalRequired', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                metadata: { loginChallengeId: 'c', deviceId: 'd' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_REQUIRED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginApprovalRequired(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_REQUIRED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalPassed', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                knownDeviceId: 'k',
                metadata: { loginChallengeId: 'c' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_PASSED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginApprovalPassed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_PASSED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalFailed', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                metadata: { loginChallengeId: 'c', failureReason: 'f' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_FAILED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginApprovalFailed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_FAILED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalExpired', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                metadata: { loginChallengeId: 'c' },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_EXPIRED,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginApprovalExpired(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_EXPIRED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalResent', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                ipAddress: 'ip',
                userAgent: 'ua',
                metadata: {
                    oldLoginChallengeId: 'old-c',
                    newLoginChallengeId: 'new-c',
                    deviceId: 'd',
                },
            };
            const event = buildMockEvent({
                ...data,
                eventType: SecurityEventType.LOGIN_APPROVAL_RESENT,
            });
            repository.save.mockResolvedValue(event);
            const result = await service.recordLoginApprovalResent(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_RESENT },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordLoginApprovalResendFailed as best-effort', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                ipAddress: 'ip',
                userAgent: 'ua',
                metadata: {
                    oldLoginChallengeId: 'old-c',
                    deviceId: 'd',
                    failureReason: 'rate_limited_cooldown',
                },
            };
            repository.save.mockRejectedValue(new Error('DB Outage'));

            const result = await service.recordLoginApprovalResendFailed(data);

            expect(result).toBeNull();
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED },
                undefined,
            );
        });

        it('should correctly wrap recordReauthPassed', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { actionScope: 'a' },
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.RE_AUTH_PASSED });
            repository.save.mockResolvedValue(event);
            const result = await service.recordReauthPassed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.RE_AUTH_PASSED },
                undefined,
            );
            expect(result).toEqual(event);
        });

        it('should correctly wrap recordReauthFailed', async () => {
            const data = {
                userId: 'u',
                realm: 'customer' as const,
                sessionId: 's',
                metadata: { actionScope: 'a', failureReason: 'f' },
            };
            const event = buildMockEvent({ ...data, eventType: SecurityEventType.RE_AUTH_FAILED });
            repository.save.mockResolvedValue(event);
            const result = await service.recordReauthFailed(data);
            expect(repository.save).toHaveBeenCalledWith(
                { ...data, eventType: SecurityEventType.RE_AUTH_FAILED },
                undefined,
            );
            expect(result).toEqual(event);
        });
    });
});
