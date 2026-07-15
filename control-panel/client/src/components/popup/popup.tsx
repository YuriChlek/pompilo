'use client';

import styles from '@/components/popup/styles.module.css';
import { ReactNode } from 'react';

type PopupProps = {
    open: boolean;
    onClose: () => void;
    title?: string;
    children: ReactNode;
};

export const Popup = ({ open, onClose, title, children }: PopupProps) => {
    if (!open) return null;

    return (
        <div className={styles.modalRoot}>
            <div className={styles.backdrop} onClick={onClose} />

            <div
                className={styles.modal}
                role="dialog"
                aria-modal="true"
                aria-labelledby="modal-title"
            >
                <header className={styles.header}>
                    {title && <h2 id="modal-title">{title}</h2>}

                    <button className={styles.close} onClick={onClose} aria-label="Close modal">
                        ✕
                    </button>
                </header>

                <div className={styles.body}>{children}</div>
            </div>
        </div>
    );
};
