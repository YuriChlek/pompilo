import type { AuthRealm } from '@/module-auth/enums/auth.enums';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import type { SecurityEventInsert } from '@/module-auth-token/schemas/security-events.schema';

type SecurityEventWriteBase = Omit<
    SecurityEventInsert,
    'eventType' | 'realm' | 'metadata' | 'createdAt'
> & {
    realm: AuthRealm;
    createdAt?: Date;
};

type SecurityEventMetadata = Record<string, unknown>;

type DeviceSessionSecurityEventType =
    | SecurityEventType.LOGIN_SUCCESS
    | SecurityEventType.REGISTRATION_SUCCESS
    | SecurityEventType.LOGIN_APPROVAL_PASSED;

type SessionSecurityEventType =
    | SecurityEventType.LOGOUT
    | SecurityEventType.SESSION_REVOKED
    | SecurityEventType.REVOKE_OTHER_SESSIONS
    | SecurityEventType.PASSWORD_CHANGED
    | SecurityEventType.EMAIL_CHANGE_REQUESTED
    | SecurityEventType.RE_AUTH_PASSED
    | SecurityEventType.RE_AUTH_FAILED;

type UserSecurityEventType =
    | SecurityEventType.REVOKE_ALL_SESSIONS
    | SecurityEventType.PASSWORD_RESET_COMPLETED
    | SecurityEventType.EMAIL_VERIFIED
    | SecurityEventType.SUSPICIOUS_DEVICE
    | SecurityEventType.LOGIN_APPROVAL_REQUIRED
    | SecurityEventType.LOGIN_APPROVAL_FAILED
    | SecurityEventType.LOGIN_APPROVAL_EXPIRED
    | SecurityEventType.LOGIN_APPROVAL_RESENT
    | SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED;

type PreSessionSecurityEventType =
    | SecurityEventType.LOGIN_FAILED
    | SecurityEventType.REFRESH_FAILED;

type ExpandSecurityEventType<T, V> = V extends any ? T & { eventType: V } : never;

export type SecurityEventWriteInput =
    | ExpandSecurityEventType<
          SecurityEventWriteBase & {
              userId: string;
              sessionId: string;
              knownDeviceId: string;
              metadata?: SecurityEventMetadata | null;
          },
          DeviceSessionSecurityEventType
      >
    | ExpandSecurityEventType<
          SecurityEventWriteBase & {
              userId: string;
              sessionId: string;
              metadata?: SecurityEventMetadata | null;
          },
          SessionSecurityEventType
      >
    | ExpandSecurityEventType<
          SecurityEventWriteBase & {
              userId: string;
              metadata: SecurityEventMetadata;
          },
          UserSecurityEventType
      >
    | ExpandSecurityEventType<
          SecurityEventWriteBase & {
              userId?: string | null;
              sessionId?: null;
              knownDeviceId?: null;
              metadata: SecurityEventMetadata;
          },
          PreSessionSecurityEventType
      >;
