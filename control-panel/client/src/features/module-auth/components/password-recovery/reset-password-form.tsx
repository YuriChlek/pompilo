'use client';

import { FormEvent, Suspense, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Button } from '@/components/button/button';
import { useResetPassword } from '@/features/module-auth/hooks/mutation';
import styles from '@/features/module-auth/components/auth-forms/styles.module.css';

const ResetPasswordFormContent = () => {
    const searchParams = useSearchParams();
    const token = searchParams.get('token') || '';
    const resetPasswordMutation = useResetPassword();
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(null);

        if (!token) {
            setError('Reset token is missing.');
            return;
        }

        if (newPassword !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        resetPasswordMutation.mutate(
            { token, newPassword },
            {
                onError: err => {
                    setError(err instanceof Error ? err.message : 'Failed to reset password.');
                },
            },
        );
    };

    return (
        <div className={styles.formWrapper}>
            <div className={styles.header}>
                <h2 className={styles.title}>Set new password</h2>
                <p className={styles.subtitle}>
                    Choose a new password for your account.
                </p>
            </div>

            <form className={styles.form} onSubmit={handleSubmit}>
                {error && <p className={styles.errorText}>{error}</p>}
                <div className={styles.inputGroup}>
                    <label className={styles.label}>New password</label>
                    <input
                        type="password"
                        name="newPassword"
                        placeholder="Minimum 8 characters"
                        required
                        className={styles.input}
                        value={newPassword}
                        onChange={event => setNewPassword(event.target.value)}
                    />
                </div>
                <div className={styles.inputGroup}>
                    <label className={styles.label}>Confirm password</label>
                    <input
                        type="password"
                        name="confirmPassword"
                        placeholder="Repeat new password"
                        required
                        className={styles.input}
                        value={confirmPassword}
                        onChange={event => setConfirmPassword(event.target.value)}
                    />
                </div>
                <Button
                    type="submit"
                    className={styles.submitButton}
                    disabled={resetPasswordMutation.isPending}
                >
                    {resetPasswordMutation.isPending ? 'Saving...' : 'Reset password'}
                </Button>
                <p className={styles.switchAuth}>
                    <Link href="/login" className={styles.link}>
                        Back to login
                    </Link>
                </p>
            </form>
        </div>
    );
};

export const ResetPasswordForm = () => (
    <Suspense fallback={<div className={styles.formWrapper}>Loading...</div>}>
        <ResetPasswordFormContent />
    </Suspense>
);
