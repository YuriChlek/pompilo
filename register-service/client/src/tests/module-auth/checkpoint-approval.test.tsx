import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { useResendCheckpoint, useVerifyCheckpoint } from '@/features/module-auth/hooks/mutation';
import { CheckpointApproval } from '@/features/module-auth/components/checkpoint-approval/checkpoint-approval';

vi.mock('@/features/module-auth/hooks/mutation', () => ({
    useVerifyCheckpoint: vi.fn(),
    useResendCheckpoint: vi.fn(),
}));

describe('CheckpointApproval', () => {
    const mockMutate = vi.fn();
    const mockResendMutate = vi.fn();
    const mockCancel = vi.fn();
    const mockCheckpointUpdated = vi.fn();
    const checkpointProps = {
        checkpointToken: 'token-123',
        loginChallengeId: 'challenge-123',
        expiresInSeconds: 300,
        resendAvailableInSeconds: 60,
        onCheckpointUpdated: mockCheckpointUpdated,
        onCancel: mockCancel,
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useVerifyCheckpoint).mockReturnValue({
            mutate: mockMutate,
            error: null,
            isPending: false,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useVerifyCheckpoint>);
        vi.mocked(useResendCheckpoint).mockReturnValue({
            mutate: mockResendMutate,
            error: null,
            isPending: false,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useResendCheckpoint>);
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('renders security checkpoint headers and digits inputs', () => {
        render(<CheckpointApproval {...checkpointProps} />);

        expect(screen.getByText('Security Checkpoint')).toBeDefined();
        expect(screen.getByText(/We've detected a login attempt/)).toBeDefined();
        expect(screen.getByText('Code expires in 5:00.')).toBeDefined();
        
        const inputs = screen.getAllByRole('textbox');
        expect(inputs.length).toBe(6);
    });

    it('submits code when all digits are filled', () => {
        render(<CheckpointApproval {...checkpointProps} />);

        const inputs = screen.getAllByRole('textbox');
        fireEvent.change(inputs[0], { target: { value: '1' } });
        fireEvent.change(inputs[1], { target: { value: '2' } });
        fireEvent.change(inputs[2], { target: { value: '3' } });
        fireEvent.change(inputs[3], { target: { value: '4' } });
        fireEvent.change(inputs[4], { target: { value: '5' } });
        fireEvent.change(inputs[5], { target: { value: '6' } });

        fireEvent.click(screen.getByText('Verify Code'));

        expect(mockMutate).toHaveBeenCalledWith(
            { checkpointToken: 'token-123', code: '123456', role: undefined },
            expect.any(Object),
        );
    });

    it('shows invalid code error state', () => {
        vi.mocked(useVerifyCheckpoint).mockReturnValue({
            mutate: mockMutate,
            error: new Error('Invalid or expired login code.'),
            isPending: false,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useVerifyCheckpoint>);

        render(<CheckpointApproval {...checkpointProps} />);

        expect(screen.getByText(/The code you entered is incorrect/)).toBeDefined();
    });

    it('shows expired challenge error state', () => {
        vi.mocked(useVerifyCheckpoint).mockReturnValue({
            mutate: mockMutate,
            error: new Error('Invalid or expired login challenge.'),
            isPending: false,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useVerifyCheckpoint>);

        render(<CheckpointApproval {...checkpointProps} />);

        expect(screen.getByText(/This security code has expired/)).toBeDefined();
    });

    it('disables resend during cooldown and enables it after the countdown', () => {
        vi.useFakeTimers();
        render(<CheckpointApproval {...checkpointProps} resendAvailableInSeconds={2} />);

        expect(screen.getByText('Resend code in 0:02')).toBeDisabled();

        act(() => {
            vi.advanceTimersByTime(2000);
        });

        expect(screen.getByText('Resend code')).toBeEnabled();
    });

    it('disables resend while verification is submitting', () => {
        vi.mocked(useVerifyCheckpoint).mockReturnValue({
            mutate: mockMutate,
            error: null,
            isPending: true,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useVerifyCheckpoint>);

        render(<CheckpointApproval {...checkpointProps} resendAvailableInSeconds={0} />);

        expect(screen.getByText('Resend code')).toBeDisabled();
    });

    it('clears entered digits and reports the new checkpoint after resend success', () => {
        mockResendMutate.mockImplementation((_input, options) => {
            options.onSuccess({
                checkpointRequired: true,
                loginChallengeId: 'new-challenge-123',
                checkpointToken: 'new-token-123',
                expiresInSeconds: 300,
                resendAvailableInSeconds: 60,
            });
        });

        render(<CheckpointApproval {...checkpointProps} resendAvailableInSeconds={0} />);

        const inputs = screen.getAllByRole('textbox') as HTMLInputElement[];
        fireEvent.change(inputs[0], { target: { value: '1' } });
        fireEvent.change(inputs[1], { target: { value: '2' } });

        fireEvent.click(screen.getByText('Resend code'));

        expect(mockResendMutate).toHaveBeenCalledWith(
            { checkpointToken: 'token-123', role: undefined },
            expect.objectContaining({
                onSuccess: expect.any(Function),
            }),
        );
        expect(inputs[0].value).toBe('');
        expect(inputs[1].value).toBe('');
        expect(screen.getByText('New code sent.')).toBeDefined();
        expect(mockCheckpointUpdated).toHaveBeenCalledWith({
            checkpointRequired: true,
            loginChallengeId: 'new-challenge-123',
            checkpointToken: 'new-token-123',
            expiresInSeconds: 300,
            resendAvailableInSeconds: 60,
        });
    });

    it('uses the replacement token for verification after resend updates props', () => {
        const { rerender } = render(
            <CheckpointApproval {...checkpointProps} resendAvailableInSeconds={0} />,
        );

        rerender(
            <CheckpointApproval
                {...checkpointProps}
                checkpointToken="new-token-123"
                loginChallengeId="new-challenge-123"
                resendAvailableInSeconds={60}
            />,
        );

        const inputs = screen.getAllByRole('textbox');
        fireEvent.change(inputs[0], { target: { value: '1' } });
        fireEvent.change(inputs[1], { target: { value: '2' } });
        fireEvent.change(inputs[2], { target: { value: '3' } });
        fireEvent.change(inputs[3], { target: { value: '4' } });
        fireEvent.change(inputs[4], { target: { value: '5' } });
        fireEvent.change(inputs[5], { target: { value: '6' } });

        fireEvent.click(screen.getByText('Verify Code'));

        expect(mockMutate).toHaveBeenCalledWith(
            { checkpointToken: 'new-token-123', code: '123456', role: undefined },
            expect.any(Object),
        );
    });

    it('shows resend error without closing the checkpoint UI', () => {
        vi.mocked(useResendCheckpoint).mockReturnValue({
            mutate: mockResendMutate,
            error: new Error('Too many verification code requests.'),
            isPending: false,
            reset: vi.fn(),
        } as unknown as ReturnType<typeof useResendCheckpoint>);

        render(<CheckpointApproval {...checkpointProps} resendAvailableInSeconds={0} />);

        expect(screen.getByText('Security Checkpoint')).toBeDefined();
        expect(screen.getByText('Too many verification code requests.')).toBeDefined();
    });

    it('shows too many attempts error state', async () => {
        render(<CheckpointApproval {...checkpointProps} />);

        const inputs = screen.getAllByRole('textbox');
        const submitButton = screen.getByText('Verify Code');

        // Trigger onError via mutating 5 times
        for (let i = 0; i < 5; i++) {
            fireEvent.change(inputs[0], { target: { value: '1' } });
            fireEvent.change(inputs[1], { target: { value: '2' } });
            fireEvent.change(inputs[2], { target: { value: '3' } });
            fireEvent.change(inputs[3], { target: { value: '4' } });
            fireEvent.change(inputs[4], { target: { value: '5' } });
            fireEvent.change(inputs[5], { target: { value: '6' } });

            // Set error mock to simulate failed verification
            vi.mocked(useVerifyCheckpoint).mockReturnValue({
                mutate: mockMutate.mockImplementation((_, options) => {
                    options.onError(new Error('Invalid or expired login code.'));
                }),
                error: new Error('Invalid or expired login code.'),
                isPending: false,
                reset: vi.fn(),
            } as unknown as ReturnType<typeof useVerifyCheckpoint>);

            fireEvent.click(submitButton);
        }

        expect(screen.getByText(/You have reached the maximum number of attempts/)).toBeDefined();
        expect(submitButton).toBeDisabled();
    });

    it('calls onCancel when back to login button is clicked', () => {
        render(<CheckpointApproval {...checkpointProps} />);

        fireEvent.click(screen.getByText('Back to Login'));
        expect(mockCancel).toHaveBeenCalled();
    });
});
