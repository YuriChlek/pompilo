'use client';

import styles from './styles.module.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import Link from 'next/link';

export const RegistrationSuccessStep = () => {
    return (
        <div className={`${styles.stepContent} ${styles.centered}`}>
            <div className={styles.successIconWrapper}>
                <FontAwesomeIcon icon={faCheck} />
            </div>

            <h2 className={styles.stepTitle}>Done!</h2>
            <p className={styles.stepDescription}>
                Your account has been created. Verify your email address before continuing.
            </p>

            <Link href="/verify-email" className={styles.nextButton}>
                Verify Email
            </Link>
        </div>
    );
};
