'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEnvelope, faCheckCircle, faExclamationCircle } from '@fortawesome/free-solid-svg-icons';
import { useVerifyEmail, useResendVerification } from '@/features/module-auth/hooks/mutation';
import styles from './styles.module.css';
import Link from 'next/link';

const EmailVerificationContent = () => {
    const searchParams = useSearchParams();
    const router = useRouter();
    const token = searchParams.get('token');

    const verifyEmailMutation = useVerifyEmail();
    const resendVerificationMutation = useResendVerification();
    const { mutate: verifyEmail } = verifyEmailMutation;

    const [resendSuccess, setResendSuccess] = useState<string | null>(null);
    const [resendError, setResendError] = useState<string | null>(null);
    const [cooldown, setCooldown] = useState<number>(0);

    // Run verification automatically if token is present
    useEffect(() => {
        if (token) {
            verifyEmail(token);
        }
    }, [token, verifyEmail]);

    // Handle resend email cooldown timer
    useEffect(() => {
        if (cooldown > 0) {
            const timer = setTimeout(() => setCooldown(cooldown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [cooldown]);

    const handleResend = () => {
        if (cooldown > 0) return;
        setResendSuccess(null);
        setResendError(null);

        resendVerificationMutation.mutate(undefined, {
            onSuccess: () => {
                setResendSuccess('Verification email has been sent successfully!');
                setCooldown(60); // 1 minute cooldown
            },
            onError: (error: unknown) => {
                const message = error instanceof Error
                    ? error.message
                    : 'Failed to resend verification email.';
                setResendError(message);
            },
        });
    };

    // Case 1: Verifying active token
    if (token) {
        if (verifyEmailMutation.isPending) {
            return (
                <div className={styles.card}>
                    <div className={styles.spinner} />
                    <h2 className={styles.title}>Verifying your email</h2>
                    <p className={styles.description}>
                        Please wait while we verify your email address. This will only take a moment.
                    </p>
                </div>
            );
        }

        if (verifyEmailMutation.isSuccess) {
            return (
                <div className={styles.card}>
                    <div className={styles.iconContainer} style={{ background: '#4caf50', boxShadow: '0 0 20px rgba(76, 175, 80, 0.4)' }}>
                        <FontAwesomeIcon icon={faCheckCircle} size="2x" />
                    </div>
                    <h2 className={styles.title}>Email Verified!</h2>
                    <p className={styles.description}>
                        Your email address has been successfully verified. You can now access all platform features.
                    </p>
                    <div className={styles.buttonContainer}>
                        <button onClick={() => router.push('/login')} className={styles.primaryButton}>
                            Go to Login
                        </button>
                    </div>
                </div>
            );
        }

        return (
            <div className={styles.card}>
                <div className={styles.iconContainer} style={{ background: '#f44336', boxShadow: '0 0 20px rgba(244, 67, 54, 0.4)' }}>
                    <FontAwesomeIcon icon={faExclamationCircle} size="2x" />
                </div>
                <h2 className={styles.title}>Verification Failed</h2>
                <p className={styles.description}>
                    {verifyEmailMutation.error?.message || 'The verification link is invalid or has expired.'}
                </p>
                <div className={styles.buttonContainer}>
                    <Link href="/login" className={styles.secondaryButton}>
                        Back to Login
                    </Link>
                </div>
            </div>
        );
    }

    // Case 2: Verification Pending screen (after registration)
    return (
        <div className={styles.card}>
            <div className={styles.iconContainer}>
                <FontAwesomeIcon icon={faEnvelope} size="2x" />
            </div>
            <h2 className={styles.title}>Verify your email</h2>
            <p className={styles.description}>
                We have sent a verification link to your email. Please check your inbox and click the link to activate your account.
            </p>

            <div className={styles.buttonContainer}>
                <button
                    onClick={handleResend}
                    disabled={resendVerificationMutation.isPending || cooldown > 0}
                    className={styles.primaryButton}
                >
                    {resendVerificationMutation.isPending
                        ? 'Sending...'
                        : cooldown > 0
                        ? `Resend available in ${cooldown}s`
                        : 'Resend email'}
                </button>
                <Link href="/login" className={styles.secondaryButton}>
                    Back to Login
                </Link>
            </div>

            {resendSuccess && <p className={styles.successText}>{resendSuccess}</p>}
            {resendError && <p className={styles.errorText}>{resendError}</p>}
        </div>
    );
};

export const EmailVerification = () => (
    <Suspense fallback={
        <div className={styles.card}>
            <div className={styles.spinner} />
            <h2 className={styles.title}>Loading...</h2>
        </div>
    }>
        <EmailVerificationContent />
    </Suspense>
);
