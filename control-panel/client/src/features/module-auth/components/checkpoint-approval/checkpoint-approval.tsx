'use client';

import { FormEvent, useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import { useResendCheckpoint, useVerifyCheckpoint } from '@/features/module-auth/hooks/mutation';
import { Button } from '@/components/button/button';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { CheckpointResponse } from '@/features/module-auth/interfaces/auth.interfaces';
import styles from '@/features/module-auth/components/auth-forms/styles.module.css';

interface CheckpointApprovalProps {
    checkpointToken: string;
    loginChallengeId: string;
    expiresInSeconds: number;
    resendAvailableInSeconds: number;
    role?: UserRoles;
    onCheckpointUpdated: (checkpoint: CheckpointResponse) => void;
    onCancel: () => void;
}

export const CheckpointApproval = ({
    checkpointToken,
    expiresInSeconds,
    resendAvailableInSeconds,
    role,
    onCheckpointUpdated,
    onCancel,
}: CheckpointApprovalProps) => {
    const { mutate, error, isPending } = useVerifyCheckpoint();
    const {
        mutate: resendCheckpoint,
        error: resendError,
        isPending: isResendPending,
    } = useResendCheckpoint();
    const [code, setCode] = useState<string[]>(Array(6).fill(''));
    const [localAttempts, setLocalAttempts] = useState(0);
    const [timerState, setTimerState] = useState({
        checkpointToken,
        expiresRemaining: expiresInSeconds,
        resendRemaining: resendAvailableInSeconds,
    });
    const [resendStatus, setResendStatus] = useState<string | null>(null);
    const inputsRef = useRef<HTMLInputElement[]>([]);
    const activeTimerState =
        timerState.checkpointToken === checkpointToken
            ? timerState
            : {
                  checkpointToken,
                  expiresRemaining: expiresInSeconds,
                  resendRemaining: resendAvailableInSeconds,
              };

    const isMaxAttempts = localAttempts >= 5;
    const errorMessage = error?.message || '';
    const expiresRemaining = activeTimerState.expiresRemaining;
    const resendRemaining = activeTimerState.resendRemaining;
    const hasCodeExpired = expiresRemaining <= 0;
    const isExpired = hasCodeExpired || errorMessage.toLowerCase().includes('expired login challenge') || errorMessage.toLowerCase().includes('invalid or expired login challenge');
    const isInvalidCode = errorMessage.toLowerCase().includes('invalid or expired login code');
    const isResendCoolingDown = resendRemaining > 0;
    const isResendDisabled = isPending || isResendPending || isResendCoolingDown;
    const resendErrorMessage = resendError?.message || '';

    const formatSeconds = (seconds: number) => {
        const safeSeconds = Math.max(0, seconds);
        const minutes = Math.floor(safeSeconds / 60);
        const remainingSeconds = safeSeconds % 60;

        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    // Focus first input on mount
    useEffect(() => {
        if (inputsRef.current[0]) {
            inputsRef.current[0].focus();
        }
    }, []);

    useEffect(() => {
        const intervalId = window.setInterval(() => {
            setTimerState(prev => {
                if (prev.checkpointToken !== checkpointToken) {
                    return {
                        checkpointToken,
                        expiresRemaining: Math.max(0, expiresInSeconds - 1),
                        resendRemaining: Math.max(0, resendAvailableInSeconds - 1),
                    };
                }

                return {
                    ...prev,
                    expiresRemaining: Math.max(0, prev.expiresRemaining - 1),
                    resendRemaining: Math.max(0, prev.resendRemaining - 1),
                };
            });
        }, 1000);

        return () => window.clearInterval(intervalId);
    }, [checkpointToken, expiresInSeconds, resendAvailableInSeconds]);

    const handleChange = (value: string, index: number) => {
        const newCode = [...code];
        newCode[index] = value.slice(-1);
        setCode(newCode);

        // Move focus to next input if filled
        if (value && index < 5 && inputsRef.current[index + 1]) {
            inputsRef.current[index + 1].focus();
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, index: number) => {
        if (e.key === 'Backspace' && !code[index] && index > 0 && inputsRef.current[index - 1]) {
            inputsRef.current[index - 1].focus();
        }
    };

    const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
        e.preventDefault();
        const pastedData = e.clipboardData.getData('text').trim();
        if (/^\d{6}$/.test(pastedData)) {
            const digits = pastedData.split('');
            setCode(digits);
            inputsRef.current[5]?.focus();
        }
    };

    const handleResend = () => {
        if (isResendDisabled) return;

        setResendStatus(null);
        resendCheckpoint(
            { checkpointToken, role },
            {
                onSuccess: checkpoint => {
                    setCode(Array(6).fill(''));
                    setLocalAttempts(0);
                    setResendStatus('New code sent.');
                    onCheckpointUpdated(checkpoint);
                },
            },
        );
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (isMaxAttempts || isExpired) return;

        const codeString = code.join('');
        if (codeString.length !== 6) return;

        mutate(
            { checkpointToken, code: codeString, role },
            {
                onError: () => {
                    setLocalAttempts(prev => prev + 1);
                },
            },
        );
    };

    return (
        <div className={styles.formWrapper}>
            <div className={styles.header}>
                <h2 className={styles.title}>Security Checkpoint</h2>
                <p className={styles.subtitle}>
                    We&apos;ve detected a login attempt that looks different from your usual activity.
                    To secure your account, we&apos;ve sent a 6-digit verification code to your email.
                </p>
                <p className={styles.subtitle}>Code expires in {formatSeconds(expiresRemaining)}.</p>
            </div>

            <form className={styles.form} onSubmit={handleSubmit}>
                <div className={clsx(styles.inputGroup, 'flex flex-col items-center gap-4')}>
                    <label className={clsx(styles.label, 'text-center')}>Enter 6-Digit Code</label>
                    <div className="flex justify-between w-full gap-2" style={{ display: 'flex', gap: '0.5rem', justifyContent: 'space-between' }}>
                        {code.map((digit, idx) => (
                            <input
                                key={idx}
                                ref={el => {
                                    if (el) inputsRef.current[idx] = el;
                                }}
                                type="text"
                                pattern="\d*"
                                maxLength={1}
                                value={digit}
                                onChange={e => handleChange(e.target.value, idx)}
                                onKeyDown={e => handleKeyDown(e, idx)}
                                onPaste={handlePaste}
                                disabled={isPending || isMaxAttempts || isExpired}
                                required
                                className={clsx(styles.input, 'text-center text-xl font-bold')}
                                style={{
                                    textAlign: 'center',
                                    fontSize: '1.5rem',
                                    padding: '0.5rem',
                                    width: '3rem',
                                    height: '3.5rem',
                                }}
                            />
                        ))}
                    </div>
                </div>

                {resendStatus && (
                    <div
                        role="status"
                        className="p-3 text-sm rounded-lg bg-green-500/10 text-green-500 border border-green-500/20"
                        style={{
                            padding: '0.75rem',
                            fontSize: '0.875rem',
                            borderRadius: '0.5rem',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            color: '#22c55e',
                            border: '1px solid rgba(34, 197, 94, 0.2)',
                            marginTop: '0.5rem',
                        }}
                    >
                        {resendStatus}
                    </div>
                )}

                {resendErrorMessage && (
                    <div
                        role="alert"
                        className="p-3 text-sm rounded-lg bg-red-500/10 text-red-500 border border-red-500/20"
                        style={{
                            padding: '0.75rem',
                            fontSize: '0.875rem',
                            borderRadius: '0.5rem',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            marginTop: '0.5rem',
                        }}
                    >
                        {resendErrorMessage}
                    </div>
                )}

                {isExpired && (
                    <div
                        className="p-3 text-sm rounded-lg bg-red-500/10 text-red-500 border border-red-500/20"
                        style={{
                            padding: '0.75rem',
                            fontSize: '0.875rem',
                            borderRadius: '0.5rem',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            marginTop: '0.5rem',
                        }}
                    >
                        <strong>Challenge Expired:</strong> This security code has expired. Request a new verification code to continue.
                    </div>
                )}

                {isMaxAttempts && (
                    <div
                        className="p-3 text-sm rounded-lg bg-red-500/10 text-red-500 border border-red-500/20"
                        style={{
                            padding: '0.75rem',
                            fontSize: '0.875rem',
                            borderRadius: '0.5rem',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            marginTop: '0.5rem',
                        }}
                    >
                        <strong>Too Many Attempts:</strong> You have reached the maximum number of attempts. This challenge has been locked for safety. Please try logging in again.
                    </div>
                )}

                {!isMaxAttempts && !isExpired && isInvalidCode && (
                    <div
                        className="p-3 text-sm rounded-lg bg-red-500/10 text-red-500 border border-red-500/20"
                        style={{
                            padding: '0.75rem',
                            fontSize: '0.875rem',
                            borderRadius: '0.5rem',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            color: '#ef4444',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            marginTop: '0.5rem',
                        }}
                    >
                        <strong>Invalid Code:</strong> The code you entered is incorrect. Remaining attempts: {5 - localAttempts}.
                    </div>
                )}

                <div className="flex flex-col gap-2 mt-4" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
                    <Button
                        type="submit"
                        disabled={isPending || isMaxAttempts || isExpired || code.join('').length !== 6}
                        className={styles.submitButton}
                    >
                        {isPending ? 'Verifying...' : 'Verify Code'}
                    </Button>
                    <Button
                        type="button"
                        variant="secondary"
                        onClick={handleResend}
                        disabled={isResendDisabled}
                        className={styles.submitButton}
                        style={{
                            backgroundColor: 'transparent',
                            border: '1px solid var(--color-app-border)',
                            color: 'var(--color-app-text-muted)',
                        }}
                    >
                        {isResendPending
                            ? 'Sending...'
                            : isResendCoolingDown
                              ? `Resend code in ${formatSeconds(resendRemaining)}`
                              : 'Resend code'}
                    </Button>
                    <Button
                        type="button"
                        variant="secondary"
                        onClick={onCancel}
                        disabled={isPending}
                        className={styles.submitButton}
                        style={{
                            backgroundColor: 'transparent',
                            border: '1px solid var(--color-app-border)',
                            color: 'var(--color-app-text-muted)',
                        }}
                    >
                        Back to Login
                    </Button>
                </div>
            </form>
        </div>
    );
};
