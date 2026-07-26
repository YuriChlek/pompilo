'use client';

import { FormEvent, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/button/button';
import { useForgotPassword } from '@/features/module-auth/hooks/mutation';
import styles from '@/features/module-auth/components/auth-forms/styles.module.css';

export const ForgotPasswordForm = () => {
    const forgotPasswordMutation = useForgotPassword();
    const [email, setEmail] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(null);

        forgotPasswordMutation.mutate(email, {
            onSuccess: () => setSubmitted(true),
            onError: err => {
                setError(err instanceof Error ? err.message : 'Failed to request password reset.');
            },
        });
    };

    return (
        <div className={styles.formWrapper}>
            <div className={styles.header}>
                <h2 className={styles.title}>Reset password</h2>
                <p className={styles.subtitle}>
                    Enter your email and we will send a one-time reset link.
                </p>
            </div>

            {submitted ? (
                <div className={styles.form}>
                    <p className={styles.subtitle}>
                        If an account exists for that email, a reset link has been sent.
                    </p>
                    <Link href="/login" className={styles.link}>
                        Back to login
                    </Link>
                </div>
            ) : (
                <form className={styles.form} onSubmit={handleSubmit}>
                    {error && <p className={styles.errorText}>{error}</p>}
                    <div className={styles.inputGroup}>
                        <label className={styles.label}>Email</label>
                        <input
                            type="email"
                            name="email"
                            placeholder="Enter your email"
                            required
                            className={styles.input}
                            value={email}
                            onChange={event => setEmail(event.target.value)}
                        />
                    </div>
                    <Button
                        type="submit"
                        className={styles.submitButton}
                        disabled={forgotPasswordMutation.isPending}
                    >
                        {forgotPasswordMutation.isPending ? 'Sending...' : 'Send reset link'}
                    </Button>
                    <p className={styles.switchAuth}>
                        Remembered your password?{' '}
                        <Link href="/login" className={styles.link}>
                            Login
                        </Link>
                    </p>
                </form>
            )}
        </div>
    );
};
