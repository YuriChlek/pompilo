import {
    RiskPolicyService,
    RiskEvaluationContext,
    UserSecurityHistory,
    RISK_SCORES,
} from '@/module-auth-token/services/risk-policy.service';
import type { AuthRealm } from '@/module-auth/enums/auth.enums';

describe('RiskPolicyService', () => {
    let service: RiskPolicyService;
    const now = new Date('2026-06-23T12:00:00Z');

    beforeEach(() => {
        service = new RiskPolicyService();
    });

    describe('evaluateRisk (Table-driven tests)', () => {
        interface TestCase {
            name: string;
            context: Partial<RiskEvaluationContext>;
            history: UserSecurityHistory;
            expectedScore: number;
            expectedDecision: 'low' | 'medium' | 'high' | 'critical';
            expectedAction: 'allow' | 'alert' | 'challenge' | 'deny';
            expectedReasons: string[];
        }

        const defaultContext: RiskEvaluationContext = {
            userId: 'user-1',
            realm: 'customer' as AuthRealm,
            deviceId: 'device-1',
            currentMetadata: {
                ipAddress: '82.207.35.41',
                userAgent: 'Mozilla/5.0 Chrome/120.0',
                country: 'UA',
                region: 'Kyiv',
                city: 'Kyiv',
            },
            isEmailVerified: true,
            isPasswordRecentlyReset: false,
            deviceBindingFailed: false,
            failedLoginAttempts24h: 0,
            failedRefreshAttempts24h: 0,
        };

        const testCases: TestCase[] = [
            {
                name: 'Registration Bootstrap (returns low risk always)',
                context: {
                    isRegistrationBootstrap: true,
                },
                history: { activeDevices: [] },
                expectedScore: 0,
                expectedDecision: 'low',
                expectedAction: 'allow',
                expectedReasons: ['registration_bootstrap'],
            },
            {
                name: 'Known Trusted Device with matching UA and Location (low risk)',
                context: {},
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: new Date(now.getTime() - 1000 * 60 * 60),
                            trustExpiresAt: new Date(now.getTime() + 1000 * 60 * 60),
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                },
                expectedScore: 0,
                expectedDecision: 'low',
                expectedAction: 'allow',
                expectedReasons: [],
            },
            {
                name: 'Known Device but untrusted/expired trust (low/medium boundary)',
                context: {},
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: null, // Untrusted
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                },
                expectedScore: RISK_SCORES.UNTRUSTED_EXISTING_DEVICE, // 15
                expectedDecision: 'low',
                expectedAction: 'allow',
                expectedReasons: ['untrusted_existing_device'],
            },
            {
                name: 'First credential login on new device/location (medium risk)',
                context: {},
                history: {
                    activeDevices: [], // No active devices => new device, new UA, new location
                },
                expectedScore: RISK_SCORES.NEW_DEVICE, // 20 (new device). Since activeDevices is empty, UA and Location don't add to score (first registration/login context)
                expectedDecision: 'low', // 20 is below MEDIUM threshold of 25
                expectedAction: 'allow',
                expectedReasons: ['new_device', 'first_credential_login'],
            },
            {
                name: 'First login of existing user on new device with existing device history (medium risk)',
                context: {
                    deviceId: 'new-device-id',
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-old',
                            trustedAt: new Date(),
                            trustExpiresAt: null,
                            lastIpAddress: '1.1.1.1',
                            lastUserAgent: 'Mozilla/5.0 Firefox',
                            lastCountry: 'UA',
                            lastRegion: 'Lviv',
                            lastCity: 'Lviv',
                        },
                    ],
                },
                expectedScore:
                    RISK_SCORES.NEW_DEVICE + // 20
                    RISK_SCORES.NEW_USER_AGENT + // 10
                    RISK_SCORES.NEW_LOCATION, // 10 => 40
                expectedDecision: 'medium', // 40 is between 25 and 50
                expectedAction: 'alert',
                expectedReasons: ['new_device', 'new_user_agent', 'new_location'],
            },
            {
                name: 'High risk (impossible travel) with verified email',
                context: {
                    currentMetadata: {
                        ipAddress: '203.0.113.195',
                        userAgent: 'Mozilla/5.0 Chrome/120.0',
                        country: 'FR',
                        region: 'Paris',
                        city: 'Paris',
                    },
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: new Date(),
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                    lastLoginEvent: {
                        timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 2), // 2 hours ago
                        country: 'UA',
                        city: 'Kyiv',
                    },
                },
                expectedScore: RISK_SCORES.IMPOSSIBLE_TRAVEL + RISK_SCORES.NEW_LOCATION, // 40 + 10 = 50
                expectedDecision: 'high',
                expectedAction: 'challenge',
                expectedReasons: ['new_location', 'impossible_travel'],
            },
            {
                name: 'High risk (impossible travel) escalated to critical if email challenge is unavailable',
                context: {
                    isEmailVerified: false,
                    currentMetadata: {
                        ipAddress: '203.0.113.195',
                        userAgent: 'Mozilla/5.0 Chrome/120.0',
                        country: 'FR',
                        region: 'Paris',
                        city: 'Paris',
                    },
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: new Date(),
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                    lastLoginEvent: {
                        timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 2), // 2 hours ago
                        country: 'UA',
                        city: 'Kyiv',
                    },
                },
                expectedScore: RISK_SCORES.IMPOSSIBLE_TRAVEL + RISK_SCORES.NEW_LOCATION, // 40 + 10 = 50
                expectedDecision: 'critical', // Escalated to critical
                expectedAction: 'deny',
                expectedReasons: [
                    'new_location',
                    'impossible_travel',
                    'email_challenge_unavailable_escalation',
                    'manual_recovery_required',
                ],
            },
            {
                name: 'High risk can challenge when email is reachable even if not verified',
                context: {
                    isEmailVerified: false,
                    hasVerifiedOrReachableEmail: true,
                    currentMetadata: {
                        ipAddress: '203.0.113.195',
                        userAgent: 'Mozilla/5.0 Chrome/120.0',
                        country: 'FR',
                        region: 'Paris',
                        city: 'Paris',
                    },
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: new Date(),
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                    lastLoginEvent: {
                        timestamp: new Date(now.getTime() - 1000 * 60 * 60 * 2),
                        country: 'UA',
                        city: 'Kyiv',
                    },
                },
                expectedScore: RISK_SCORES.IMPOSSIBLE_TRAVEL + RISK_SCORES.NEW_LOCATION,
                expectedDecision: 'high',
                expectedAction: 'challenge',
                expectedReasons: ['new_location', 'impossible_travel'],
            },
            {
                name: 'Password recently reset on new device (high risk)',
                context: {
                    isPasswordRecentlyReset: true,
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-old',
                            trustedAt: new Date(),
                            trustExpiresAt: null,
                            lastIpAddress: '1.1.1.1',
                            lastUserAgent: 'Mozilla/5.0 Firefox',
                            lastCountry: 'UA',
                            lastRegion: 'Lviv',
                            lastCity: 'Lviv',
                        },
                    ],
                },
                expectedScore:
                    RISK_SCORES.NEW_DEVICE + // 20
                    RISK_SCORES.NEW_USER_AGENT + // 10
                    RISK_SCORES.NEW_LOCATION + // 10
                    RISK_SCORES.PASSWORD_RECENTLY_RESET, // 30 => 70
                expectedDecision: 'high',
                expectedAction: 'challenge',
                expectedReasons: [
                    'new_device',
                    'new_user_agent',
                    'new_location',
                    'password_recently_reset',
                ],
            },
            {
                name: 'Critical risk: device binding failed + multiple failed logins',
                context: {
                    deviceBindingFailed: true,
                    failedLoginAttempts24h: 6,
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: null,
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                },
                expectedScore:
                    RISK_SCORES.UNTRUSTED_EXISTING_DEVICE + // 15
                    RISK_SCORES.DEVICE_BINDING_FAILED + // 40
                    RISK_SCORES.FAILED_LOGINS_HIGH, // 40 => 95
                expectedDecision: 'critical',
                expectedAction: 'deny',
                expectedReasons: [
                    'untrusted_existing_device',
                    'high_failed_login_attempts',
                    'device_binding_failed',
                ],
            },
            {
                name: 'Critical risk with unavailable email challenge requires manual recovery',
                context: {
                    isEmailVerified: false,
                    deviceBindingFailed: true,
                    failedLoginAttempts24h: 6,
                },
                history: {
                    activeDevices: [
                        {
                            deviceId: 'device-1',
                            trustedAt: null,
                            trustExpiresAt: null,
                            lastIpAddress: '82.207.35.41',
                            lastUserAgent: 'Mozilla/5.0 Chrome/120.0',
                            lastCountry: 'UA',
                            lastRegion: 'Kyiv',
                            lastCity: 'Kyiv',
                        },
                    ],
                },
                expectedScore:
                    RISK_SCORES.UNTRUSTED_EXISTING_DEVICE +
                    RISK_SCORES.DEVICE_BINDING_FAILED +
                    RISK_SCORES.FAILED_LOGINS_HIGH,
                expectedDecision: 'critical',
                expectedAction: 'deny',
                expectedReasons: [
                    'untrusted_existing_device',
                    'high_failed_login_attempts',
                    'device_binding_failed',
                    'manual_recovery_required',
                ],
            },
        ];

        testCases.forEach(
            ({
                name,
                context,
                history,
                expectedScore,
                expectedDecision,
                expectedAction,
                expectedReasons,
            }) => {
                it(name, () => {
                    const fullContext = { ...defaultContext, ...context };
                    const result = service.evaluateRisk(fullContext, history, now);

                    expect(result.score).toBe(expectedScore);
                    expect(result.decision).toBe(expectedDecision);
                    expect(result.requiredAction).toBe(expectedAction);
                    expect(result.reasons).toEqual(expect.arrayContaining(expectedReasons));
                });
            },
        );
    });
});
