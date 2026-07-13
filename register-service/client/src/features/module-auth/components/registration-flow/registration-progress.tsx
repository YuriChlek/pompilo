import styles from './styles.module.css';
import { RegistrationStep } from '@/features/module-auth/types/registration.types';

type RegistrationProgressProps = {
    currentStep: RegistrationStep;
};

export const RegistrationProgress = ({ currentStep }: RegistrationProgressProps) => {
    const visualStep = currentStep === RegistrationStep.SUCCESS ? 2 : 1;

    return (
        <div className={styles.progressIndicator}>
            <div
                className={`${styles.progressDot} ${visualStep >= 1 ? styles.active : ''}`}
            />
            <div
                className={`${styles.progressLine} ${visualStep >= 2 ? styles.active : ''}`}
            />
            <div
                className={`${styles.progressDot} ${visualStep >= 2 ? styles.active : ''}`}
            />
        </div>
    );
};
