'use client';

import { useState } from 'react';
import styles from './styles.module.css';

export const PrivacySettingsSection = () => {
    const [hideEmail, setHideEmail] = useState(true);
    const [securityAlerts, setSecurityAlerts] = useState(true);

    return (
        <div className={styles.sectionContainer} data-testid="privacy-settings-section">
            <h3 className={styles.sectionTitle}>Приватність</h3>
            <p className={styles.sectionDescription}>Керуйте видимістю ідентифікаційних даних та службовими сповіщеннями.</p>

            <div className={styles.privacyOptions}>
                <div className={styles.toggleRow}>
                    <div className={styles.toggleLabelContainer}>
                        <div className={styles.toggleTitle}>Приховати email</div>
                        <div className={styles.toggleDescription}>Не показувати email у клієнтському інтерфейсі за межами налаштувань безпеки.</div>
                    </div>
                    <button 
                        onClick={() => setHideEmail(!hideEmail)} 
                        className={`${styles.toggleSwitch} ${hideEmail ? styles.toggleActive : ''}`}
                    >
                        <div className={`${styles.toggleCircle} ${hideEmail ? styles.toggleCircleActive : ''}`} />
                    </button>
                </div>

                <div className={styles.toggleRow}>
                    <div className={styles.toggleLabelContainer}>
                        <div className={styles.toggleTitle}>Сповіщення безпеки</div>
                        <div className={styles.toggleDescription}>Отримувати повідомлення про входи, зміну пароля та критичні дії з акаунтом.</div>
                    </div>
                    <button 
                        onClick={() => setSecurityAlerts(!securityAlerts)} 
                        className={`${styles.toggleSwitch} ${securityAlerts ? styles.toggleActive : ''}`}
                    >
                        <div className={`${styles.toggleCircle} ${securityAlerts ? styles.toggleCircleActive : ''}`} />
                    </button>
                </div>
            </div>
        </div>
    );
};
