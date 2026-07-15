'use client';

import { useState } from 'react';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { useUser } from '@/features/module-auth/hooks/query';
import {
    useChangePasswordMutation,
    useChangeEmailRequestMutation,
    useChangeEmailConfirmMutation,
} from '@/features/module-account/hooks/mutation';
import { ReauthModal } from '@/features/module-auth/components/reauth-modal/reauth-modal';
import { authService } from '@/features/module-auth/api-service/client';
import styles from './styles.module.css';

type SecuritySettingsSectionProps = {
    role: UserRoles;
};

export const SecuritySettingsSection = ({ role }: SecuritySettingsSectionProps) => {
    const { data: user } = useUser(role);
    const currentEmail = user?.email || 'user@example.com';

    const [isEditingEmail, setIsEditingEmail] = useState(false);
    const [newEmail, setNewEmail] = useState('');
    const [otpCode, setOtpCode] = useState(['', '', '', '', '', '']);
    const [isEmailPending, setIsEmailPending] = useState(false);

    const [oldPassword, setOldPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');

    const [emailSuccess, setEmailSuccess] = useState('');
    const [emailError, setEmailError] = useState('');
    const [passwordSuccess, setPasswordSuccess] = useState('');
    const [passwordError, setPasswordError] = useState('');
    const [isReauthOpen, setIsReauthOpen] = useState(false);

    const changePasswordMutation = useChangePasswordMutation(role);
    const changeEmailRequestMutation = useChangeEmailRequestMutation(role);
    const changeEmailConfirmMutation = useChangeEmailConfirmMutation(role);

    const handleOtpChange = (index: number, value: string) => {
        if (value.length > 1) return;
        const newOtp = [...otpCode];
        newOtp[index] = value;
        setOtpCode(newOtp);

        if (value !== '' && index < 5) {
            const nextInput = document.getElementById(`otp-${index + 1}`);
            nextInput?.focus();
        }
    };

    const handleEmailChangeRequest = (e: React.FormEvent) => {
        e.preventDefault();
        setEmailError('');
        setEmailSuccess('');
        setIsReauthOpen(true);
    };

    const handleEmailChangeConfirm = (e: React.FormEvent) => {
        e.preventDefault();
        setEmailError('');
        setEmailSuccess('');
        const code = otpCode.join('');
        changeEmailConfirmMutation.mutate(code, {
            onSuccess: () => {
                setIsEmailPending(false);
                setOtpCode(['', '', '', '', '', '']);
                setEmailSuccess('Email успішно змінено!');
            },
            onError: (err: unknown) => {
                const msg = err instanceof Error ? err.message : 'Невірний або застарілий код';
                setEmailError(msg);
            },
        });
    };

    const handlePasswordChange = async (e: React.FormEvent) => {
        e.preventDefault();
        setPasswordError('');
        setPasswordSuccess('');

        if (newPassword !== confirmPassword) {
            setPasswordError('Новий пароль та підтвердження не збігаються');
            return;
        }

        try {
            // Under the hood reauth using already typed current password
            const reauthResult = await authService.reauth(oldPassword, 'password_change', role);

            changePasswordMutation.mutate(
                { data: { oldPassword, newPassword }, reauthConfirmationToken: reauthResult.confirmationToken },
                {
                    onSuccess: () => {
                        setOldPassword('');
                        setNewPassword('');
                        setConfirmPassword('');
                        setPasswordSuccess('Пароль успішно оновлено!');
                    },
                    onError: (err: unknown) => {
                        const msg = err instanceof Error ? err.message : 'Не вдалося змінити пароль';
                        setPasswordError(msg);
                    },
                }
            );
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Невірний поточний пароль';
            setPasswordError(msg);
        }
    };

    return (
        <div className={styles.sectionContainer} data-testid="security-settings-section">
            <h3 className={styles.sectionTitle}>Логін та безпека</h3>
            
            <div className={styles.securityBlock}>
                {/* Email Change Card */}
                <div className={`${styles.card} ${isEmailPending ? styles.pendingCard : ''}`}>
                    <div className={styles.cardHeader}>
                        <div>
                            <span className={styles.cardLabel}>Email</span>
                            <div className={styles.emailDisplay}>
                                <span className={styles.emailText}>{currentEmail}</span>
                                {!isEmailPending ? (
                                    <span className={styles.verifiedBadge}>Verified</span>
                                ) : (
                                    <span className={styles.unverifiedBadge}>Unverified</span>
                                )}
                            </div>
                        </div>
                        {!isEditingEmail && !isEmailPending && (
                            <button 
                                onClick={() => {
                                    setIsEditingEmail(true);
                                    setEmailSuccess('');
                                    setEmailError('');
                                }} 
                                className={styles.linkButton}
                            >
                                Змінити
                            </button>
                        )}
                    </div>

                    {emailSuccess && <div className={styles.successText} style={{ color: 'var(--color-app-success)', fontSize: '0.825rem', fontWeight: 600 }}>{emailSuccess}</div>}
                    {emailError && <div className={styles.errorText} style={{ color: 'var(--color-app-danger)', fontSize: '0.825rem', fontWeight: 600 }}>{emailError}</div>}

                    {isEditingEmail && (
                        <form onSubmit={handleEmailChangeRequest} className={styles.emailForm}>
                            <input
                                type="email"
                                placeholder="Новий Email"
                                value={newEmail}
                                onChange={(e) => setNewEmail(e.target.value)}
                                className={styles.inputField}
                                required
                            />
                            <div className={styles.buttonGroup}>
                                <button 
                                    type="submit" 
                                    className={styles.primaryButton}
                                    disabled={changeEmailRequestMutation.isPending}
                                >
                                    {changeEmailRequestMutation.isPending ? 'Надсилання...' : 'Надіслати код'}
                                </button>
                                <button 
                                    type="button" 
                                    onClick={() => setIsEditingEmail(false)} 
                                    className={styles.secondaryButton}
                                >
                                    Скасувати
                                </button>
                            </div>
                        </form>
                    )}

                    {isEmailPending && (
                        <div className={styles.otpContainer}>
                            <p className={styles.otpPrompt}>Введіть 6-значний код з пошти:</p>
                            <form onSubmit={handleEmailChangeConfirm} className={styles.otpForm}>
                                <div className={styles.otpInputs}>
                                    {otpCode.map((digit, idx) => (
                                        <input
                                            key={idx}
                                            id={`otp-${idx}`}
                                            type="text"
                                            maxLength={1}
                                            value={digit}
                                            onChange={(e) => handleOtpChange(idx, e.target.value)}
                                            className={styles.otpInput}
                                        />
                                    ))}
                                </div>
                                <div className={styles.buttonGroup}>
                                    <button 
                                        type="submit" 
                                        className={styles.primaryButton}
                                        disabled={changeEmailConfirmMutation.isPending}
                                    >
                                        {changeEmailConfirmMutation.isPending ? 'Перевірка...' : 'Підтвердити'}
                                    </button>
                                    <button 
                                        type="button" 
                                        onClick={() => {
                                            setIsEmailPending(false);
                                            setEmailSuccess('');
                                            setEmailError('');
                                        }}
                                        className={styles.secondaryButton}
                                    >
                                        Скасувати
                                    </button>
                                </div>
                            </form>
                        </div>
                    )}
                </div>

                {/* Password Change Card */}
                <div className={styles.card}>
                    <h4 className={styles.cardTitle}>Зміна пароля</h4>
                    
                    {passwordSuccess && <div style={{ color: 'var(--color-app-success)', fontSize: '0.825rem', fontWeight: 600 }}>{passwordSuccess}</div>}
                    {passwordError && <div style={{ color: 'var(--color-app-danger)', fontSize: '0.825rem', fontWeight: 600 }}>{passwordError}</div>}

                    <form onSubmit={handlePasswordChange} className={styles.passwordForm}>
                        <div className={styles.formField}>
                            <label htmlFor="oldPassword" className={styles.fieldLabel}>Поточний пароль</label>
                            <input
                                id="oldPassword"
                                type="password"
                                value={oldPassword}
                                onChange={(e) => setOldPassword(e.target.value)}
                                className={styles.inputField}
                                required
                            />
                        </div>
                        <div className={styles.formField}>
                            <label htmlFor="newPassword" className={styles.fieldLabel}>Новий пароль</label>
                            <input
                                id="newPassword"
                                type="password"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                className={styles.inputField}
                                required
                            />
                        </div>
                        <div className={styles.formField}>
                            <label htmlFor="confirmPassword" className={styles.fieldLabel}>Підтвердження нового пароля</label>
                            <input
                                id="confirmPassword"
                                type="password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                className={styles.inputField}
                                required
                            />
                        </div>
                        <button 
                            type="submit" 
                            className={styles.primaryButton}
                            disabled={changePasswordMutation.isPending}
                        >
                            {changePasswordMutation.isPending ? 'Збереження...' : 'Оновити пароль'}
                        </button>
                    </form>
                </div>
            </div>

            <ReauthModal
                isOpen={isReauthOpen}
                onClose={() => setIsReauthOpen(false)}
                onSuccess={(token) => {
                    changeEmailRequestMutation.mutate(
                        { newEmail, reauthConfirmationToken: token },
                        {
                            onSuccess: () => {
                                setIsEmailPending(true);
                                setIsEditingEmail(false);
                                setEmailSuccess('Код підтвердження надіслано на нову пошту');
                            },
                            onError: (err: unknown) => {
                                const msg = err instanceof Error ? err.message : 'Помилка надсилання запиту';
                                setEmailError(msg);
                            },
                        }
                    );
                }}
                actionScope="email_change"
                role={role}
            />
        </div>
    );
};
