import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { Mock } from 'vitest';
import { RegistrationFlow } from '@/features/module-auth/components/registration-flow/registration-flow';
import { useRegister } from '@/features/module-auth/hooks/mutation';

vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: vi.fn(),
        refresh: vi.fn(),
        replace: vi.fn(),
    }),
    usePathname: () => '/',
}));

vi.mock('@/features/module-auth/hooks/mutation', () => ({
    useRegister: vi.fn(),
}));

describe('RegistrationFlow generic identity UI', () => {
    const mockRegister = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();

        (useRegister as Mock).mockReturnValue({
            mutateAsync: mockRegister,
            isPending: false,
        });
    });

    const fillRegistrationForm = () => {
        fireEvent.change(screen.getByPlaceholderText('e.g. John Doe'), {
            target: { value: 'John Doe' },
        });
        fireEvent.change(screen.getByPlaceholderText('alex@example.com'), {
            target: { value: 'john@example.com' },
        });
        fireEvent.change(screen.getByPlaceholderText('Minimum 8 characters'), {
            target: { value: 'password123' },
        });
    };

    it('renders a generic registration form without domain role choices', () => {
        render(<RegistrationFlow />);

        expect(screen.getByText('Create Account')).toBeDefined();
        expect(screen.queryByText('I am an Athlete')).toBeNull();
        expect(screen.queryByText('I am a Coach')).toBeNull();
        expect(screen.queryByText(/Date of Birth/i)).toBeNull();
        expect(screen.queryByText(/Professional Settings/i)).toBeNull();
    });

    it('submits only the generic identity payload', async () => {
        mockRegister.mockResolvedValue({ id: 'user-1', role: 'user' });

        render(<RegistrationFlow />);
        fillRegistrationForm();
        fireEvent.click(screen.getByRole('button', { name: /Continue/i }));

        await waitFor(() => {
            expect(mockRegister).toHaveBeenCalledWith({
                name: 'John Doe',
                email: 'john@example.com',
                password: 'password123',
            });
        });
    });

    it('shows verify email success CTA after generic registration', async () => {
        mockRegister.mockResolvedValue({ id: 'user-1', role: 'user' });

        render(<RegistrationFlow />);
        fillRegistrationForm();
        fireEvent.click(screen.getByRole('button', { name: /Continue/i }));

        await waitFor(() => {
            expect(screen.getByText('Done!')).toBeDefined();
        });

        const cta = screen.getByRole('link', { name: /Verify Email/i });
        expect(cta.getAttribute('href')).toBe('/verify-email');
    });

    it('shows an error and stays on the generic form when registration fails', async () => {
        mockRegister.mockRejectedValue(new Error('Email already exists'));

        render(<RegistrationFlow />);
        fillRegistrationForm();
        fireEvent.click(screen.getByRole('button', { name: /Continue/i }));

        await waitFor(() => {
            expect(screen.getByText('Email already exists')).toBeDefined();
        });

        expect(screen.getByText('Create Account')).toBeDefined();
        expect(screen.queryByText('Done!')).toBeNull();
    });
});
