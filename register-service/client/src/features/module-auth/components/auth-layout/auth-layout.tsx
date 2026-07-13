'use client';

import Image from 'next/image';
import styles from './styles.module.css';

interface AuthLayoutProps {
    children: React.ReactNode;
}

export const AuthLayout = ({ children }: AuthLayoutProps) => {
    return (
        <div className={styles.authContainer}>
            <div className={styles.imageSection}>
                <Image
                    className={styles.heroImage}
                    src="https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1470&auto=format&fit=crop"
                    alt="Gym"
                    fill
                    unoptimized
                />
                <div className={styles.imageContent}>
                    <div className={styles.logoP}>P</div>
                    <h1 className={styles.brandName}>
                        Pom<span className={styles.brandAccent}>pilo</span>
                    </h1>
                    <p className={styles.description}>
                        A complete ecosystem for reaching your training goals.
                    </p>
                </div>
            </div>
            <div className={styles.formSection}>
                <div className={styles.mobileHeader}>
                    <div className={styles.mobileLogo}>P</div>
                    <h1 className={styles.mobileBrandName}>
                        Pom<span className={styles.brandAccent}>pilo</span>
                    </h1>
                </div>
                {children}
            </div>
        </div>
    );
};
