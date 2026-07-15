'use client';

import { useState } from 'react';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { useSessionsQuery } from '@/features/module-account/hooks/query';
import {
    useRevokeSessionMutation,
    useRevokeOtherSessionsMutation,
    useLogoutAllSessionsMutation,
} from '@/features/module-account/hooks/mutation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMobileScreen, faLaptop, faSpinner } from '@fortawesome/free-solid-svg-icons';
import { ReauthModal } from '@/features/module-auth/components/reauth-modal/reauth-modal';
import styles from './styles.module.css';

type SessionsSettingsSectionProps = {
    role: UserRoles;
};

export const SessionsSettingsSection = ({ role }: SessionsSettingsSectionProps) => {
    const { data: sessions = [], isLoading, error } = useSessionsQuery(role);

    const revokeSessionMutation = useRevokeSessionMutation(role);
    const revokeOtherSessionsMutation = useRevokeOtherSessionsMutation(role);
    const logoutAllSessionsMutation = useLogoutAllSessionsMutation(role);

    const [isReauthOpen, setIsReauthOpen] = useState(false);

    const handleRevokeSession = (id: string) => {
        revokeSessionMutation.mutate(id);
    };

    const handleRevokeOtherSessions = () => {
        if (confirm('Ви впевнені, що хочете завершити всі інші сесії?')) {
            setIsReauthOpen(true);
        }
    };

    const handleLogoutAllSessions = () => {
        if (confirm('Ви впевнені, що хочете завершити всі сесії? Вам потрібно буде увійти знову.')) {
            logoutAllSessionsMutation.mutate();
        }
    };

    if (isLoading) {
        return (
            <div className={styles.loading}>
                <FontAwesomeIcon icon={faSpinner} spin className={styles.infoIcon} style={{ marginRight: '0.5rem' }} />
                <span>Завантаження сесій...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.errorCard}>
                <span className={styles.errorTitle}>Помилка</span>
                <span className={styles.errorText}>Не вдалося завантажити активні сесії.</span>
            </div>
        );
    }

    return (
        <div className={styles.sectionContainer} data-testid="sessions-settings-section">
            <h3 className={styles.sectionTitle}>Активні сесії</h3>
            <p className={styles.sectionDescription}>Пристрої, на яких виконано вхід у ваш акаунт.</p>

            <div className={styles.sessionsList}>
                {sessions.map(session => {
                    const isCurrent = session.currentSession;
                    const ua = session.userAgent?.toLowerCase() || '';
                    const isIphone = ua.includes('iphone') || ua.includes('mobile') || ua.includes('android');
                    
                    // Simple clean browser/OS extraction
                    let deviceName = 'Browser';
                    if (ua.includes('macintosh')) deviceName = 'Mac';
                    else if (ua.includes('windows')) deviceName = 'Windows PC';
                    else if (ua.includes('iphone')) deviceName = 'iPhone';
                    else if (ua.includes('ipad')) deviceName = 'iPad';
                    else if (ua.includes('android')) deviceName = 'Android Device';
                    else if (ua.includes('linux')) deviceName = 'Linux PC';

                    const isTrusted = Boolean(session.trustedAt);
                    const isSuspicious = typeof session.riskScore === 'number' && session.riskScore > 0.5;
                    const isUntrusted = !isTrusted;

                    return (
                        <div 
                            key={session.id} 
                            className={`${styles.sessionItem} ${isCurrent ? styles.currentSessionItem : ''}`}
                        >
                            <div className={styles.sessionLeft}>
                                <div className={`${styles.deviceIcon} ${isCurrent ? styles.currentDeviceIcon : ''}`}>
                                    <FontAwesomeIcon icon={isIphone ? faMobileScreen : faLaptop} />
                                </div>
                                <div className={styles.sessionInfo}>
                                    <div className={styles.deviceName}>
                                        {deviceName}
                                        {isCurrent && <span className={styles.currentBadge}>Цей</span>}
                                        {isTrusted && <span className={styles.trustedBadge}>Довірений</span>}
                                        {isSuspicious && <span className={styles.suspiciousBadge}>Підозрілий</span>}
                                        {isUntrusted && !isCurrent && <span className={styles.untrustedBadge}>Невідомий</span>}
                                    </div>
                                    <div className={styles.sessionMeta}>
                                        {session.ipAddress || 'Unknown IP'}
                                        {session.approximateLocation ? ` (${session.approximateLocation})` : ''} • {new Date(session.lastSeenAt ?? session.createdAt).toLocaleString('uk-UA')}
                                    </div>
                                </div>
                            </div>
                            <button 
                                onClick={() => handleRevokeSession(session.id)} 
                                className={styles.revokeButton}
                                disabled={revokeSessionMutation.isPending}
                            >
                                {isCurrent ? 'Вийти з цієї сесії' : 'Вийти'}
                            </button>
                        </div>
                    );
                })}
            </div>

            {sessions.length > 1 && (
                <button 
                    onClick={handleRevokeOtherSessions} 
                    className={styles.revokeOtherButton}
                    disabled={revokeOtherSessionsMutation.isPending}
                >
                    {revokeOtherSessionsMutation.isPending ? 'Вихід...' : 'Завершити інші сесії'}
                </button>
            )}

            {sessions.length > 0 && (
                <button
                    onClick={handleLogoutAllSessions}
                    className={styles.revokeOtherButton}
                    disabled={logoutAllSessionsMutation.isPending}
                >
                    {logoutAllSessionsMutation.isPending ? 'Вихід...' : 'Завершити всі сесії'}
                </button>
            )}

            <ReauthModal
                isOpen={isReauthOpen}
                onClose={() => setIsReauthOpen(false)}
                onSuccess={(token) => {
                    revokeOtherSessionsMutation.mutate(token);
                }}
                actionScope="revoke_other_sessions"
                role={role}
            />
        </div>
    );
};
