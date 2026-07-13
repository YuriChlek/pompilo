'use client';

import { FormEvent, useState } from 'react';
import clsx from 'clsx';
import styles from '@/features/module-auth/components/auth-forms/styles.module.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEye, faEyeSlash } from '@fortawesome/free-solid-svg-icons';
import { faGoogle, faApple } from '@fortawesome/free-brands-svg-icons';
import { useLogin } from '@/features/module-auth/hooks/mutation';
import { Button } from '@/components/button/button';
import { CheckpointApproval } from '@/features/module-auth/components/checkpoint-approval/checkpoint-approval';
import type { LoginFormProps } from '@/features/module-auth/interfaces/component-props.interfaces';
import type { CheckpointResponse } from '@/features/module-auth/interfaces/auth.interfaces';
import type { LoginData } from '@/features/module-auth/types/auth.types';

import Link from 'next/link';

export const LoginForm = ({ mode, title, variant = 'modern' }: LoginFormProps) => {
    const { mutate, reset } = useLogin();
    const [checkpointState, setCheckpointState] = useState<CheckpointResponse | null>(null);
    const [showPassword, setShowPassword] = useState(false);
    const isModern = variant === 'modern';

    if (checkpointState?.checkpointRequired) {
        return (
            <CheckpointApproval
                checkpointToken={checkpointState.checkpointToken}
                loginChallengeId={checkpointState.loginChallengeId}
                expiresInSeconds={checkpointState.expiresInSeconds}
                resendAvailableInSeconds={checkpointState.resendAvailableInSeconds}
                role={mode}
                onCheckpointUpdated={setCheckpointState}
                onCancel={() => {
                    setCheckpointState(null);
                    reset();
                }}
            />
        );
    }

    function handleSubmit(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();

        const formData = new FormData(event.currentTarget);
        const login = formData.get('login')?.toString() || '';
        const password = formData.get('password')?.toString() || '';
        const loginData: LoginData = {
            login,
            password,
            ...(mode ? { role: mode } : {}),
        };

        mutate(loginData, {
            onSuccess: result => {
                if (result && 'checkpointRequired' in result && result.checkpointRequired) {
                    setCheckpointState(result);
                }
            },
        });
    }

    return (
        <div className={clsx(styles.formWrapper, !isModern && styles.formWrapperCompact)}>
            {isModern && (
                <div className={styles.header}>
                    <h2 className={styles.title}>Welcome back!</h2>
                    <p className={styles.subtitle}>Log in to your account to continue.</p>
                </div>
            )}
            <form
                className={clsx(styles.form, !isModern && styles.formCompact)}
                onSubmit={handleSubmit}
            >
                {!isModern && title && <h2 className={styles.compactTitle}>{title}</h2>}
                <div className={styles.inputGroup}>
                    {isModern && <label className={styles.label}>Email</label>}
                    <input
                        type={isModern ? 'email' : 'text'}
                        name="login"
                        placeholder={isModern ? 'Enter your email' : 'Login'}
                        required
                        className={styles.input}
                    />
                </div>
                <div className={styles.inputGroup}>
                    <div className={styles.labelWrapper}>
                        {isModern && <label className={styles.label}>Password</label>}
                        {isModern && (
                            <Link href="/forgot-password" className={styles.forgotLink}>
                                Forgot password?
                            </Link>
                        )}
                    </div>
                    <div className={styles.passwordField}>
                        <input
                            type={showPassword ? 'text' : 'password'}
                            name="password"
                            placeholder={isModern ? 'Enter your password' : 'Password'}
                            required
                            className={styles.input}
                        />
                        <button
                            type="button"
                            className={styles.passwordToggle}
                            onClick={() => setShowPassword(prev => !prev)}
                            aria-label={showPassword ? 'Hide password' : 'Show password'}
                        >
                            <FontAwesomeIcon
                                icon={showPassword ? faEyeSlash : faEye}
                                width={20}
                                height={20}
                            />
                        </button>
                    </div>
                </div>
                <Button type="submit" className={styles.submitButton}>
                    Login
                </Button>
            </form>

            {isModern && (
                <>
                    <div className={styles.divider}>
                        <div className={styles.line}></div>
                        <span className={styles.dividerText}>or</span>
                        <div className={styles.line}></div>
                    </div>

                    <div className={styles.socialButtons}>
                        <button type="button" className={styles.socialButton}>
                            <FontAwesomeIcon icon={faGoogle} className={styles.googleIcon} />
                            Google
                        </button>
                        <button type="button" className={styles.socialButton}>
                            <FontAwesomeIcon icon={faApple} className={styles.appleIcon} />
                            Apple
                        </button>
                    </div>

                    <p className={styles.switchAuth}>
                        Do not have an account?{' '}
                        <Link href="/register" className={styles.link}>
                            Create one now
                        </Link>
                    </p>
                </>
            )}
        </div>
    );
};
