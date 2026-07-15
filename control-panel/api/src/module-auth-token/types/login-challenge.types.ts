export type LoginChallengeResendEligibleState = 'active' | 'expired';

export type LoginChallengeResendIneligibleReason =
    | 'consumed'
    | 'failed'
    | 'superseded'
    | 'outside_resend_window';

export type LoginChallengeResendEligibility =
    | {
          eligible: true;
          state: LoginChallengeResendEligibleState;
          resendWindowExpiresAt: Date;
      }
    | {
          eligible: false;
          reason: LoginChallengeResendIneligibleReason;
          resendWindowExpiresAt: Date;
      };

export interface LoginChallengeResendPolicyInput {
    userId: string;
    realm: string;
    deviceId: string;
    ipAddress: string;
}

export type LoginChallengeResendPolicyLimitReason =
    | 'cooldown'
    | 'user_device_rate_limited'
    | 'ip_rate_limited';

export type LoginChallengeResendPolicyResult =
    | {
          allowed: true;
          retryAfterSeconds: 0;
      }
    | {
          allowed: false;
          reason: LoginChallengeResendPolicyLimitReason;
          retryAfterSeconds: number;
      };
