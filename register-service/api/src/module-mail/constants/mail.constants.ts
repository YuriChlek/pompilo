export const MAIL_SERVICE = Symbol.for('MAIL_SERVICE');
export const MAIL_QUEUE = 'mail';

export const MAIL_POISON_FAILURE_CODES = {
    DECRYPTION_FAILURE: 'decryption_failure',
    INVALID_PAYLOAD_SCHEMA: 'invalid_payload_schema',
    PLAINTEXT_PAYLOAD_FORBIDDEN: 'plaintext_payload_forbidden',
} as const;

export type MailPoisonFailureCode =
    (typeof MAIL_POISON_FAILURE_CODES)[keyof typeof MAIL_POISON_FAILURE_CODES];
