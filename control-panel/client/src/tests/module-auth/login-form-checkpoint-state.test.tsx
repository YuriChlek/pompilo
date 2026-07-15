import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { LoginForm } from '@/features/module-auth/components/auth-forms/login-form';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { CheckpointResponse } from '@/features/module-auth/interfaces/auth.interfaces';

const mocks = vi.hoisted(() => ({
    mutate: vi.fn(),
    reset: vi.fn(),
    checkpointProps: [] as Array<{
        checkpointToken: string;
        loginChallengeId: string;
        expiresInSeconds: number;
        resendAvailableInSeconds: number;
        role?: UserRoles;
        onCheckpointUpdated: (checkpoint: CheckpointResponse) => void;
        onCancel: () => void;
    }>,
}));

vi.mock('@/features/module-auth/hooks/mutation', () => ({
    useLogin: () => ({
        mutate: mocks.mutate,
        reset: mocks.reset,
    }),
}));

vi.mock('@/features/module-auth/components/checkpoint-approval/checkpoint-approval', () => ({
    CheckpointApproval: (props: (typeof mocks.checkpointProps)[number]) => {
        mocks.checkpointProps.push(props);

        return (
            <div data-testid="checkpoint-approval">
                <span>{props.checkpointToken}</span>
                <span>{props.loginChallengeId}</span>
                <button
                    type="button"
                    onClick={() =>
                        props.onCheckpointUpdated({
                            checkpointRequired: true,
                            loginChallengeId: 'new-challenge-id',
                            checkpointToken: 'new-checkpoint-token',
                            expiresInSeconds: 300,
                            resendAvailableInSeconds: 60,
                        })
                    }
                >
                    Simulate resend success
                </button>
                <button type="button" onClick={props.onCancel}>
                    Cancel checkpoint
                </button>
            </div>
        );
    },
}));

describe('LoginForm checkpoint state ownership', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.checkpointProps.length = 0;
    });

    it('stores checkpoint identity in the parent and replaces it after resend success', () => {
        render(<LoginForm mode={UserRoles.PLATFORM_ADMIN} title="Admin Login" variant="compact" />);

        fireEvent.change(screen.getByPlaceholderText('Login'), {
            target: { value: 'admin@example.com' },
        });
        fireEvent.change(screen.getByPlaceholderText('Password'), {
            target: { value: 'Password123' },
        });
        fireEvent.click(screen.getByText('Login'));

        expect(mocks.mutate).toHaveBeenCalledWith(
            {
                login: 'admin@example.com',
                password: 'Password123',
                role: UserRoles.PLATFORM_ADMIN,
            },
            expect.objectContaining({
                onSuccess: expect.any(Function),
            }),
        );

        const initialCheckpoint: CheckpointResponse = {
            checkpointRequired: true,
            loginChallengeId: 'old-challenge-id',
            checkpointToken: 'old-checkpoint-token',
            expiresInSeconds: 120,
            resendAvailableInSeconds: 30,
        };
        const mutateOptions = mocks.mutate.mock.calls[0]?.[1] as {
            onSuccess: (result: CheckpointResponse) => void;
        };

        act(() => {
            mutateOptions.onSuccess(initialCheckpoint);
        });

        expect(screen.getByText('old-checkpoint-token')).toBeDefined();
        expect(mocks.checkpointProps.at(-1)).toMatchObject({
            checkpointToken: 'old-checkpoint-token',
            loginChallengeId: 'old-challenge-id',
            expiresInSeconds: 120,
            resendAvailableInSeconds: 30,
            role: UserRoles.PLATFORM_ADMIN,
        });

        fireEvent.click(screen.getByText('Simulate resend success'));

        expect(screen.getByText('new-checkpoint-token')).toBeDefined();
        expect(mocks.checkpointProps.at(-1)).toMatchObject({
            checkpointToken: 'new-checkpoint-token',
            loginChallengeId: 'new-challenge-id',
            expiresInSeconds: 300,
            resendAvailableInSeconds: 60,
            role: UserRoles.PLATFORM_ADMIN,
        });
    });
});
