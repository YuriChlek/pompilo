'use client';

import { FormEvent, useState } from 'react';
import { authService } from '@/features/module-auth/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { Button } from '@/components/button/button';

interface ReauthModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess: (token: string) => void;
    actionScope: string;
    role: UserRoles;
}

export const ReauthModal = ({
    isOpen,
    onClose,
    onSuccess,
    actionScope,
    role,
}: ReauthModalProps) => {
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isPending, setIsPending] = useState(false);

    if (!isOpen) return null;

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError('');
        setIsPending(true);

        try {
            const result = await authService.reauth(password, actionScope, role);
            if (result && result.confirmationToken) {
                setPassword('');
                onSuccess(result.confirmationToken);
                onClose();
            } else {
                setError('Не вдалося отримати токен підтвердження');
            }
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Невірний пароль або помилка авторизації';
            setError(msg);
        } finally {
            setIsPending(false);
        }
    };

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: 'rgba(0, 0, 0, 0.6)',
                backdropFilter: 'blur(8px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 9999,
                animation: 'fadeIn 0.2s ease-out',
            }}
        >
            <div
                style={{
                    backgroundColor: 'var(--color-app-surface, #1e1e24)',
                    border: '1px solid var(--color-app-border, #2d2d34)',
                    borderRadius: '1rem',
                    padding: '2rem',
                    width: '100%',
                    maxWidth: '26rem',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem',
                }}
            >
                <div>
                    <h3
                        style={{
                            fontSize: '1.25rem',
                            fontWeight: 700,
                            color: 'var(--color-app-text, #ffffff)',
                            marginBottom: '0.5rem',
                        }}
                    >
                        Потрібне підтвердження
                    </h3>
                    <p
                        style={{
                            fontSize: '0.875rem',
                            color: 'var(--color-app-text-muted, #a0a0b0)',
                            lineHeight: 1.5,
                        }}
                    >
                        Для виконання цієї дії безпеки введіть свій поточний пароль для продовження.
                    </p>
                </div>

                <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                        <label
                            htmlFor="reauth-password"
                            style={{
                                fontSize: '0.875rem',
                                fontWeight: 600,
                                color: 'var(--color-app-text, #ffffff)',
                            }}
                        >
                            Пароль
                        </label>
                        <input
                            id="reauth-password"
                            type="password"
                            required
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            disabled={isPending}
                            placeholder="Введіть ваш пароль"
                            style={{
                                width: '100%',
                                padding: '0.75rem',
                                borderRadius: '0.75rem',
                                border: '1px solid var(--color-app-border, #2d2d34)',
                                backgroundColor: 'var(--color-app-bg, #121214)',
                                color: 'var(--color-app-text, #ffffff)',
                                outline: 'none',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>

                    {error && (
                        <div
                            style={{
                                fontSize: '0.825rem',
                                color: '#ef4444',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                padding: '0.75rem',
                                borderRadius: '0.5rem',
                                border: '1px solid rgba(239, 68, 68, 0.2)',
                                fontWeight: 600,
                            }}
                        >
                            {error}
                        </div>
                    )}

                    <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
                        <Button
                            type="submit"
                            disabled={isPending || !password}
                            style={{ flex: 1 }}
                        >
                            {isPending ? 'Підтвердження...' : 'Підтвердити'}
                        </Button>
                        <Button
                            type="button"
                            variant="secondary"
                            disabled={isPending}
                            onClick={onClose}
                            style={{
                                flex: 1,
                                backgroundColor: 'transparent',
                                border: '1px solid var(--color-app-border, #2d2d34)',
                                color: 'var(--color-app-text-muted, #a0a0b0)',
                            }}
                        >
                            Скасувати
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
};
