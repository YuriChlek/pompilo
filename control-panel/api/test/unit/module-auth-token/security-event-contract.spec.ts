import { SECURITY_EVENT_CONTRACTS } from '@/module-auth-token/constants/security-event.constants';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';

describe('SecurityEventContracts', () => {
    it('defines purpose, required fields and retention for every event type', () => {
        expect(Object.keys(SECURITY_EVENT_CONTRACTS).toSorted()).toEqual(
            Object.values(SecurityEventType).toSorted(),
        );

        for (const contract of Object.values(SECURITY_EVENT_CONTRACTS)) {
            expect(contract.purpose.length).toBeGreaterThan(0);
            expect(contract.requiredFields).toContain('realm');
            expect(contract.retentionClass).toMatch(/^(authentication|account-security)$/);
        }
    });

    it('defines resend audit metadata without sensitive code values', () => {
        expect(SECURITY_EVENT_CONTRACTS[SecurityEventType.LOGIN_APPROVAL_RESENT]).toMatchObject({
            requiredFields: ['userId', 'realm'],
            requiredMetadataFields: ['oldLoginChallengeId', 'newLoginChallengeId', 'deviceId'],
        });
        expect(
            SECURITY_EVENT_CONTRACTS[SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED],
        ).toMatchObject({
            requiredFields: ['userId', 'realm'],
            requiredMetadataFields: ['oldLoginChallengeId', 'deviceId', 'failureReason'],
        });

        const metadataFields = [
            ...SECURITY_EVENT_CONTRACTS[SecurityEventType.LOGIN_APPROVAL_RESENT]
                .requiredMetadataFields,
            ...SECURITY_EVENT_CONTRACTS[SecurityEventType.LOGIN_APPROVAL_RESEND_FAILED]
                .requiredMetadataFields,
        ];

        expect(metadataFields).not.toContain('code');
        expect(metadataFields).not.toContain('checkpointToken');
        expect(metadataFields).not.toContain('codeHash');
        expect(metadataFields).not.toContain('checkpointTokenHash');
    });
});
