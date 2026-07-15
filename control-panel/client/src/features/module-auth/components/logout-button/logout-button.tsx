'use client';

import clsx from 'clsx';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/button/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRightFromBracket, faRightToBracket } from '@fortawesome/free-solid-svg-icons';
import { useLogout } from '@/features/module-auth/hooks/mutation';
import styles from '@/features/module-auth/components/logout-button/styles.module.css';
import type { LogoutButtonProps } from '@/features/module-auth/interfaces/component-props.interfaces';

export const LogoutButton = ({
    isAuthenticated,
    role,
    loginPath = '/login',
    className,
    forceCompact,
}: LogoutButtonProps) => {
    const router = useRouter();
    const { mutate, isPending } = useLogout();

    const handleClick = () => {
        if (isAuthenticated) {
            mutate(role);

            return;
        }

        router.push(loginPath);
    };

    const label = isAuthenticated ? 'Log out' : 'Log in';
    const ariaLabel = isAuthenticated ? 'Log out from your account' : 'Navigate to login page';
    const icon = isAuthenticated ? faRightFromBracket : faRightToBracket;

    return (
        <Button
            className={clsx(styles.authButton, className, { [styles.forceCompact]: forceCompact })}
            variant="ghost"
            size="sm"
            type="button"
            onClick={handleClick}
            disabled={isPending}
            aria-label={ariaLabel}
        >
            <div className={styles.icon}>
                <FontAwesomeIcon icon={icon} />
            </div>
            {!forceCompact && <span className={styles.label}>{label}</span>}
        </Button>
    );
};
