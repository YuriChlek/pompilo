'use client';

import styles from './styles.module.css';
import { Button } from '@/components/button/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faArrowRight,
    faSpinner,
} from '@fortawesome/free-solid-svg-icons';
import Link from 'next/link';
import { useState, useEffect } from 'react';

type BasicData = {
    name: string;
    email: string;
    password: string;
};

type RegistrationBasicStepProps = {
    onSubmit: () => void;
    initialData: BasicData;
    onDataChange: (data: BasicData) => void;
    isSubmitting: boolean;
    error: string | null;
};

export const RegistrationBasicStep = ({
    onSubmit,
    initialData,
    onDataChange,
    isSubmitting,
    error,
}: RegistrationBasicStepProps) => {
    const [name, setName] = useState(initialData.name);
    const [email, setEmail] = useState(initialData.email);
    const [password, setPassword] = useState(initialData.password);

    // Sync local state to parent when navigating away or changing
    useEffect(() => {
        onDataChange({ name, email, password });
    }, [name, email, password, onDataChange]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit();
    };

    return (
        <div className={styles.stepContent}>
            <h2 className={styles.stepTitle}>Create Account</h2>
            <p className={styles.stepDescription}>Enter your identity details to create your account.</p>

            {error && <div className={styles.errorMessage}>{error}</div>}

            <form className={styles.stepForm} onSubmit={handleSubmit}>
                <div className={styles.fieldGroup}>
                    <label className={styles.label}>Full Name</label>
                    <input
                        type="text"
                        placeholder="e.g. John Doe"
                        className={styles.input}
                        value={name}
                        onChange={e => setName(e.target.value)}
                        required
                    />
                </div>

                <div className={styles.fieldGroup}>
                    <label className={styles.label}>Email</label>
                    <input
                        type="email"
                        placeholder="alex@example.com"
                        className={styles.input}
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        required
                    />
                </div>

                <div className={styles.fieldGroup}>
                    <label className={styles.label}>Password</label>
                    <input
                        type="password"
                        placeholder="Minimum 8 characters"
                        className={styles.input}
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        required
                    />
                </div>

                <Button type="submit" className={styles.nextButton} disabled={isSubmitting}>
                    {isSubmitting ? (
                        <>
                            <FontAwesomeIcon icon={faSpinner} spin /> Please wait...
                        </>
                    ) : (
                        <>
                            Continue <FontAwesomeIcon icon={faArrowRight} />
                        </>
                    )}
                </Button>
            </form>

            <p className={styles.switchAuth}>
                Already have an account?{' '}
                <Link href="/login" className={styles.link}>
                    Login
                </Link>
            </p>
        </div>
    );
};
