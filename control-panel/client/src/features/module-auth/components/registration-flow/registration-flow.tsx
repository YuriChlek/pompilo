'use client';

import { useState } from 'react';
import styles from './styles.module.css';
import { RegistrationBasicStep } from './registration-basic-step';
import { RegistrationSuccessStep } from './registration-success-step';
import { RegistrationProgress } from './registration-progress';
import { RegistrationStep } from '@/features/module-auth/types/registration.types';
import { useRegister } from '@/features/module-auth/hooks/mutation';

export const RegistrationFlow = () => {
    const [step, setStep] = useState<RegistrationStep>(RegistrationStep.BASIC_INFO);

    const [basicData, setBasicData] = useState({
        name: '',
        email: '',
        password: '',
    });

    const [status, setStatus] = useState<'idle' | 'submitting' | 'success'>('idle');
    const [error, setError] = useState<string | null>(null);

    const { mutateAsync: register } = useRegister(false);

    const handleBasicSubmit = async () => {
        setError(null);
        setStatus('submitting');

        try {
            await register(basicData);

            setStatus('success');
            setStep(RegistrationStep.SUCCESS);
        } catch (err: unknown) {
            setStatus('idle');
            const message = err instanceof Error ? err.message : 'Registration failed. Please try again.';
            setError(message);
        }
    };

    return (
        <div className={styles.formWrapper}>
            <RegistrationProgress currentStep={step} />

            {step === RegistrationStep.BASIC_INFO && (
                <RegistrationBasicStep
                    onSubmit={handleBasicSubmit}
                    initialData={basicData}
                    onDataChange={setBasicData}
                    isSubmitting={status === 'submitting'}
                    error={step === RegistrationStep.BASIC_INFO ? error : null}
                />
            )}

            {step === RegistrationStep.SUCCESS && <RegistrationSuccessStep />}
        </div>
    );
};
