'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import {
    useDeactivateAccountMutation,
    useScheduleAccountDeletionMutation,
} from '@/features/module-account/hooks/mutation';
import { ReauthModal } from '@/features/module-auth/components/reauth-modal/reauth-modal';
import styles from './styles.module.css';

type DangerZoneSectionProps = {
    role: UserRoles;
};

export const DangerZoneSection = ({ role }: DangerZoneSectionProps) => {
    const router = useRouter();
    const deactivateMutation = useDeactivateAccountMutation(role);
    const deleteMutation = useScheduleAccountDeletionMutation(role);

    const [isReauthOpen, setIsReauthOpen] = useState(false);
    const [reauthAction, setReauthAction] = useState<'deactivate' | 'delete' | null>(null);

    const handleDeactivate = () => {
        if (confirm('Ви впевнені, що хочете деактивувати свій аккаунт? Вас буде розлоговано.')) {
            setReauthAction('deactivate');
            setIsReauthOpen(true);
        }
    };

    const handleDelete = () => {
        if (confirm('Ця дія запланує видалення акаунта через 30 днів. Ви впевнені?')) {
            setReauthAction('delete');
            setIsReauthOpen(true);
        }
    };

    const handleReauthSuccess = (token: string) => {
        if (reauthAction === 'deactivate') {
            deactivateMutation.mutate(token, {
                onSuccess: () => {
                    alert('Аккаунт деактивовано. Дякуємо, що були з нами!');
                    router.push('/');
                },
                onError: (err: unknown) => {
                    const msg = err instanceof Error ? err.message : 'Не вдалося деактивувати аккаунт';
                    alert(msg);
                },
            });
        } else if (reauthAction === 'delete') {
            deleteMutation.mutate(token, {
                onSuccess: () => {
                    alert('Видалення акаунта заплановано. Ви можете скасувати його, увійшовши знову протягом 30 днів.');
                    router.push('/');
                },
                onError: (err: unknown) => {
                    const msg = err instanceof Error ? err.message : 'Не вдалося запланувати видалення';
                    alert(msg);
                },
            });
        }
    };

    return (
        <div className={styles.sectionContainer} data-testid="danger-zone-section">
            <h3 className={`${styles.sectionTitle} ${styles.dangerText}`}>Керування акаунтом</h3>
            <p className={styles.sectionDescription}>Деактивація або повне видалення вашого профілю.</p>

            <div className={styles.dangerOptions}>
                <div className={styles.dangerBox}>
                    <div className={styles.dangerBoxLeft}>
                        <div className={styles.dangerBoxTitle}>Тимчасова деактивація</div>
                        <div className={styles.dangerBoxDescription}>
                            Тимчасово вимкне ваш аккаунт. Ви зможете активувати його знову, виконавши вхід.
                        </div>
                    </div>
                    <button 
                        onClick={handleDeactivate} 
                        className={styles.deactivateButton}
                        disabled={deactivateMutation.isPending}
                    >
                        {deactivateMutation.isPending ? 'Вимкнення...' : 'Деактивувати'}
                    </button>
                </div>

                <div className={`${styles.dangerBox} ${styles.deletionBox}`}>
                    <div className={styles.dangerBoxLeft}>
                        <div className={styles.dangerBoxTitle}>Видалити назавжди</div>
                        <div className={styles.dangerBoxDescription}>
                            Безповоротно видалить ваші дані, історію та плани після пільгового періоду 30 днів.
                        </div>
                    </div>
                    <button 
                        onClick={handleDelete} 
                        className={styles.deleteButton}
                        disabled={deleteMutation.isPending}
                    >
                        {deleteMutation.isPending ? 'Запуск...' : 'Видалити назавжди'}
                    </button>
                </div>
            </div>

            <ReauthModal
                isOpen={isReauthOpen}
                onClose={() => {
                    setIsReauthOpen(false);
                    setReauthAction(null);
                }}
                onSuccess={handleReauthSuccess}
                actionScope={reauthAction === 'deactivate' ? 'account_deactivate' : 'account_delete'}
                role={role}
            />
        </div>
    );
};
