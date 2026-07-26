import { Injectable } from '@nestjs/common';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';

export interface RiskEvaluationContext {
    userId: string;
    realm: AuthRealm;
    deviceId: string;
    currentMetadata: {
        ipAddress: string;
        userAgent: string;
        country?: string;
        region?: string;
        city?: string;
    };
    isEmailVerified: boolean;
    isPasswordRecentlyReset: boolean;
    deviceBindingFailed: boolean;
    failedLoginAttempts24h: number;
    failedRefreshAttempts24h: number;
    isRegistrationBootstrap?: boolean;
    hasVerifiedOrReachableEmail?: boolean;
}

export interface UserSecurityHistory {
    activeDevices: {
        deviceId: string;
        trustedAt: Date | null;
        trustExpiresAt: Date | null;
        lastIpAddress?: string | null;
        lastCountry?: string | null;
        lastRegion?: string | null;
        lastCity?: string | null;
        lastUserAgent?: string | null;
    }[];
    lastLoginEvent?: {
        timestamp: Date;
        ipAddress?: string | null;
        country?: string | null;
        region?: string | null;
        city?: string | null;
    } | null;
}

export interface RiskDecision {
    score: number;
    decision: 'low' | 'medium' | 'high' | 'critical';
    reasons: string[];
    requiredAction: 'allow' | 'alert' | 'challenge' | 'deny';
}

export const RISK_SCORES = {
    NEW_DEVICE: 20,
    UNTRUSTED_EXISTING_DEVICE: 15,
    NEW_USER_AGENT: 10,
    NEW_LOCATION: 10,
    IMPOSSIBLE_TRAVEL: 40,
    FAILED_LOGINS_HIGH: 40,
    FAILED_REFRESH_HIGH: 40,
    DEVICE_BINDING_FAILED: 40,
    PASSWORD_RECENTLY_RESET: 30,
};

export const RISK_THRESHOLDS = {
    MEDIUM: 25,
    HIGH: 50,
    CRITICAL: 80,
};

@Injectable()
export class RiskPolicyService {
    public evaluateRisk(
        context: RiskEvaluationContext,
        history: UserSecurityHistory,
        now = new Date(),
    ): RiskDecision {
        const reasons: string[] = [];

        const canUseEmailChallenge = context.hasVerifiedOrReachableEmail ?? context.isEmailVerified;
        const isFirstCredentialLogin =
            history.activeDevices.length === 0 && history.lastLoginEvent == null;

        // 1. Check bootstrap registration
        if (context.isRegistrationBootstrap) {
            return {
                score: 0,
                decision: 'low',
                reasons: ['registration_bootstrap'],
                requiredAction: 'allow',
            };
        }

        let score = 0;

        // 2. Device signals
        const matchingDevice = history.activeDevices.find(d => d.deviceId === context.deviceId);

        if (!matchingDevice) {
            score += RISK_SCORES.NEW_DEVICE;
            reasons.push('new_device');
            if (isFirstCredentialLogin) {
                reasons.push('first_credential_login');
            }
        } else {
            const isTrusted =
                matchingDevice.trustedAt !== null &&
                (matchingDevice.trustExpiresAt === null ||
                    matchingDevice.trustExpiresAt.getTime() > now.getTime());

            if (!isTrusted) {
                score += RISK_SCORES.UNTRUSTED_EXISTING_DEVICE;
                reasons.push('untrusted_existing_device');
            }
        }

        // 3. User agent signals
        const hasMatchingUa = history.activeDevices.some(
            d => d.lastUserAgent === context.currentMetadata.userAgent,
        );
        if (history.activeDevices.length > 0 && !hasMatchingUa) {
            score += RISK_SCORES.NEW_USER_AGENT;
            reasons.push('new_user_agent');
        }

        // 4. Geolocation signals
        const hasMatchingLocation = history.activeDevices.some(
            d =>
                d.lastCountry === context.currentMetadata.country &&
                d.lastRegion === context.currentMetadata.region &&
                d.lastCity === context.currentMetadata.city,
        );
        if (history.activeDevices.length > 0 && !hasMatchingLocation) {
            score += RISK_SCORES.NEW_LOCATION;
            reasons.push('new_location');
        }

        // 5. Impossible travel
        if (history.lastLoginEvent) {
            const timeDiffMs = now.getTime() - new Date(history.lastLoginEvent.timestamp).getTime();
            const timeDiffHours = timeDiffMs / (1000 * 60 * 60);

            const countryChanged =
                history.lastLoginEvent.country &&
                context.currentMetadata.country &&
                history.lastLoginEvent.country !== context.currentMetadata.country;

            const cityChanged =
                history.lastLoginEvent.city &&
                context.currentMetadata.city &&
                history.lastLoginEvent.city !== context.currentMetadata.city;

            if ((countryChanged && timeDiffHours < 3) || (cityChanged && timeDiffHours < 1)) {
                score += RISK_SCORES.IMPOSSIBLE_TRAVEL;
                reasons.push('impossible_travel');
            }
        }

        // 6. Attempt thresholds
        if (context.failedLoginAttempts24h >= 5) {
            score += RISK_SCORES.FAILED_LOGINS_HIGH;
            reasons.push('high_failed_login_attempts');
        }

        if (context.failedRefreshAttempts24h >= 5) {
            score += RISK_SCORES.FAILED_REFRESH_HIGH;
            reasons.push('high_failed_refresh_attempts');
        }

        // 7. Device binding failure
        if (context.deviceBindingFailed) {
            score += RISK_SCORES.DEVICE_BINDING_FAILED;
            reasons.push('device_binding_failed');
        }

        // 8. Password reset/change
        if (context.isPasswordRecentlyReset) {
            score += RISK_SCORES.PASSWORD_RECENTLY_RESET;
            reasons.push('password_recently_reset');
        }

        // Determine decision category
        let decision: 'low' | 'medium' | 'high' | 'critical' = 'low';
        if (score >= RISK_THRESHOLDS.CRITICAL) {
            decision = 'critical';
        } else if (score >= RISK_THRESHOLDS.HIGH) {
            decision = 'high';
        } else if (score >= RISK_THRESHOLDS.MEDIUM) {
            decision = 'medium';
        }

        const hasSevereSignal =
            reasons.includes('impossible_travel') ||
            reasons.includes('high_failed_login_attempts') ||
            reasons.includes('high_failed_refresh_attempts') ||
            reasons.includes('device_binding_failed') ||
            reasons.includes('password_recently_reset');

        // Rollout bootstrap guard: first credential login without severe signals must not
        // challenge every existing dev/test user when device history is empty.
        if (isFirstCredentialLogin && !hasSevereSignal && decision === 'high') {
            decision = 'medium';
            reasons.push('first_login_safeguard_downgrade');
        }

        // Safeguard for users that cannot complete email-code approval.
        if (decision === 'high' && !canUseEmailChallenge) {
            decision = 'critical';
            reasons.push('email_challenge_unavailable_escalation');
        }

        if (decision === 'critical' && !canUseEmailChallenge) {
            reasons.push('manual_recovery_required');
        }

        // Map decision to required action
        const actionMap: Record<typeof decision, 'allow' | 'alert' | 'challenge' | 'deny'> = {
            low: 'allow',
            medium: 'alert',
            high: 'challenge',
            critical: 'deny',
        };

        return {
            score,
            decision,
            reasons,
            requiredAction: actionMap[decision],
        };
    }
}
