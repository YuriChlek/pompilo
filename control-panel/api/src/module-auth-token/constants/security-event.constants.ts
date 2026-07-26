import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';

export type SecurityEventRetentionClass = 'authentication' | 'account-security';

export interface SecurityEventContract {
    purpose: string;
    requiredFields: readonly string[];
    requiredMetadataFields: readonly string[];
    retentionClass: SecurityEventRetentionClass;
}

export const SECURITY_EVENT_CONTRACTS = {
    [SecurityEventType.LOGIN_SUCCESS]: {
        purpose: 'Records a successful credential login and issued logical session.',
        requiredFields: ['userId', 'realm', 'sessionId', 'knownDeviceId'],
        requiredMetadataFields: [],
        retentionClass: 'authentication',
    },
    [SecurityEventType.REGISTRATION_SUCCESS]: {
        purpose: 'Records successful registration and creation of the first session.',
        requiredFields: ['userId', 'realm', 'sessionId', 'knownDeviceId'],
        requiredMetadataFields: [],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_FAILED]: {
        purpose: 'Records a rejected credential login, including pre-user failures.',
        requiredFields: ['realm', 'ipAddress'],
        requiredMetadataFields: ['failureReason'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGOUT]: {
        purpose: 'Records explicit logout of the current logical session.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: [],
        retentionClass: 'authentication',
    },
    [SecurityEventType.SESSION_REVOKED]: {
        purpose: 'Records revocation of one logical session.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['revocationReason'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.REVOKE_OTHER_SESSIONS]: {
        purpose: 'Records a request that revoked all sessions except the current session.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['revokedSessionCount'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.REVOKE_ALL_SESSIONS]: {
        purpose: 'Records revocation of every session in all applicable realms.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['revokedSessionCount', 'revocationReason'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.REFRESH_FAILED]: {
        purpose: 'Records a failed refresh-token verification or binding decision.',
        requiredFields: ['realm', 'ipAddress'],
        requiredMetadataFields: ['failureReason'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.PASSWORD_CHANGED]: {
        purpose: 'Records an authenticated password change and resulting revocation.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['revokedSessionCount'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.PASSWORD_RESET_COMPLETED]: {
        purpose: 'Records completion of password reset and cross-realm revocation.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['revokedSessionCount'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.EMAIL_CHANGE_REQUESTED]: {
        purpose: 'Records creation of an email-change verification challenge.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['newEmailHash'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.EMAIL_VERIFIED]: {
        purpose: 'Records successful verification of the primary account email.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: [],
        retentionClass: 'account-security',
    },
    [SecurityEventType.SUSPICIOUS_DEVICE]: {
        purpose: 'Records a risk decision for a new or suspicious browser installation.',
        requiredFields: ['userId', 'realm', 'ipAddress'],
        requiredMetadataFields: ['deviceId', 'riskSignals'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_REQUIRED]: {
        purpose: 'Records creation of a pending login approval challenge.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['loginChallengeId', 'deviceId'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_PASSED]: {
        purpose: 'Records successful login challenge approval.',
        requiredFields: ['userId', 'realm', 'sessionId', 'knownDeviceId'],
        requiredMetadataFields: ['loginChallengeId'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_FAILED]: {
        purpose: 'Records an invalid login approval attempt.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['loginChallengeId', 'failureReason'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_EXPIRED]: {
        purpose: 'Records expiry of an unconsumed login challenge.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['loginChallengeId'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_RESENT]: {
        purpose: 'Records successful creation of a replacement login approval challenge.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['oldLoginChallengeId', 'newLoginChallengeId', 'deviceId'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED]: {
        purpose: 'Records a rejected login approval resend attempt.',
        requiredFields: ['userId', 'realm'],
        requiredMetadataFields: ['oldLoginChallengeId', 'deviceId', 'failureReason'],
        retentionClass: 'authentication',
    },
    [SecurityEventType.RE_AUTH_PASSED]: {
        purpose: 'Records successful re-authentication for a sensitive action.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['actionScope'],
        retentionClass: 'account-security',
    },
    [SecurityEventType.RE_AUTH_FAILED]: {
        purpose: 'Records rejected re-authentication for a sensitive action.',
        requiredFields: ['userId', 'realm', 'sessionId'],
        requiredMetadataFields: ['actionScope', 'failureReason'],
        retentionClass: 'account-security',
    },
} as const satisfies Record<SecurityEventType, SecurityEventContract>;
