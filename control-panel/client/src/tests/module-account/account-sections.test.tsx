import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SecuritySettingsSection } from '@/features/module-account/components/security-settings-section';
import { SessionsSettingsSection } from '@/features/module-account/components/sessions-settings-section';
import { PrivacySettingsSection } from '@/features/module-account/components/privacy-settings-section';
import { DangerZoneSection } from '@/features/module-account/components/danger-zone-section';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import * as React from 'react';

vi.mock('@/features/module-auth/api-service/client', () => ({
    authService: {
        reauth: vi.fn().mockResolvedValue({ confirmationToken: 'reauth-token-xyz' }),
    },
}));

vi.mock('@/features/module-auth/components/reauth-modal/reauth-modal', () => ({
    ReauthModal: ({ isOpen, onSuccess }: { isOpen: boolean; onSuccess: (token: string) => void }) => {
        React.useEffect(() => {
            if (isOpen) {
                onSuccess('mocked-token');
            }
        }, [isOpen, onSuccess]);
        return null;
    },
}));

// Mock mutations
const mockChangePassword = vi.fn();
const mockChangeEmailRequest = vi.fn();
const mockChangeEmailConfirm = vi.fn();
const mockDeactivateAccount = vi.fn();
const mockScheduleAccountDeletion = vi.fn();
const mockRevokeSession = vi.fn();
const mockRevokeOtherSessions = vi.fn();
const mockLogoutAllSessions = vi.fn();
const mockUseSessionsQuery = vi.fn();

const defaultSessions = [
    {
        id: 'sess-1',
        device: 'Mac',
        ipAddress: '127.0.0.1',
        createdAt: '2026-06-05T00:00:00Z',
        userAgent: 'macintosh',
        currentSession: true,
    },
    {
        id: 'sess-2',
        device: 'iPhone',
        ipAddress: '192.168.1.1',
        createdAt: '2026-06-05T00:00:00Z',
        userAgent: 'iphone',
        currentSession: false,
    },
];

vi.mock('@/features/module-auth/hooks/query', () => ({
    useUser: () => ({
        data: { email: 'alex.user@email.com' },
        isLoading: false,
    }),
}));

vi.mock('@/features/module-account/hooks/query', () => ({
    useSessionsQuery: () => mockUseSessionsQuery(),
    ACCOUNT_QUERY_KEYS: {
        sessions: () => ['sessions'],
    },
}));

vi.mock('@/features/module-account/hooks/mutation', () => ({
    useChangePasswordMutation: () => ({ mutate: mockChangePassword, isPending: false }),
    useChangeEmailRequestMutation: () => ({ mutate: mockChangeEmailRequest, isPending: false }),
    useChangeEmailConfirmMutation: () => ({ mutate: mockChangeEmailConfirm, isPending: false }),
    useDeactivateAccountMutation: () => ({ mutate: mockDeactivateAccount, isPending: false }),
    useScheduleAccountDeletionMutation: () => ({ mutate: mockScheduleAccountDeletion, isPending: false }),
    useRevokeSessionMutation: () => ({ mutate: mockRevokeSession, isPending: false }),
    useRevokeOtherSessionsMutation: () => ({ mutate: mockRevokeOtherSessions, isPending: false }),
    useLogoutAllSessionsMutation: () => ({ mutate: mockLogoutAllSessions, isPending: false }),
}));

vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: vi.fn(),
    }),
}));

describe('SecuritySettingsSection', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockUseSessionsQuery.mockReturnValue({
            data: defaultSessions,
            isLoading: false,
            error: null,
        });
    });

    it('renders current email and change password form', () => {
        render(<SecuritySettingsSection role={UserRoles.USER} />);

        expect(screen.getByText('alex.user@email.com')).toBeDefined();
        expect(screen.getByText('Verified')).toBeDefined();
        expect(screen.getByText('Зміна пароля')).toBeDefined();
    });

    it('enters email editing mode and triggers mutation request', async () => {
        render(<SecuritySettingsSection role={UserRoles.USER} />);

        const editBtn = screen.getByText('Змінити');
        fireEvent.click(editBtn);

        const emailInput = screen.getByPlaceholderText('Новий Email');
        fireEvent.change(emailInput, { target: { value: 'new@example.com' } });

        const submitBtn = screen.getByText('Надіслати код');
        fireEvent.click(submitBtn);

        expect(mockChangeEmailRequest).toHaveBeenCalledWith(
            { newEmail: 'new@example.com', reauthConfirmationToken: 'mocked-token' },
            expect.any(Object)
        );
    });

    it('triggers password change mutation', async () => {
        render(<SecuritySettingsSection role={UserRoles.USER} />);

        const oldPassInput = screen.getByLabelText('Поточний пароль');
        const newPassInput = screen.getByLabelText('Новий пароль');
        const confirmPassInput = screen.getByLabelText('Підтвердження нового пароля');

        fireEvent.change(oldPassInput, { target: { value: 'oldpass' } });
        fireEvent.change(newPassInput, { target: { value: 'newpass' } });
        fireEvent.change(confirmPassInput, { target: { value: 'newpass' } });

        const submitBtn = screen.getByText('Оновити пароль');
        fireEvent.click(submitBtn);

        await waitFor(() => {
            expect(mockChangePassword).toHaveBeenCalledWith(
                {
                    data: { oldPassword: 'oldpass', newPassword: 'newpass' },
                    reauthConfirmationToken: 'reauth-token-xyz',
                },
                expect.any(Object)
            );
        });
    });
});

describe('SessionsSettingsSection', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders sessions list and handles revocation', () => {
        render(<SessionsSettingsSection role={UserRoles.USER} />);

        expect(screen.getByText('Mac')).toBeDefined();
        expect(screen.getByText('iPhone')).toBeDefined();
        expect(screen.getByText('Цей')).toBeDefined();

        const revokeBtn = screen.getByText('Вийти');
        fireEvent.click(revokeBtn);

        expect(mockRevokeSession).toHaveBeenCalledWith('sess-2');
    });

    it('renders current-session revoke action and calls revoke for current session (Phase 29.3)', () => {
        render(<SessionsSettingsSection role={UserRoles.USER} />);

        const revokeCurrentBtn = screen.getByText('Вийти з цієї сесії');
        fireEvent.click(revokeCurrentBtn);

        expect(mockRevokeSession).toHaveBeenCalledWith('sess-1');
    });

    it('handles revoking other sessions', () => {
        window.confirm = vi.fn().mockReturnValue(true);
        render(<SessionsSettingsSection role={UserRoles.USER} />);

        const revokeOthersBtn = screen.getByText('Завершити інші сесії');
        fireEvent.click(revokeOthersBtn);

        expect(mockRevokeOtherSessions).toHaveBeenCalledWith('mocked-token');
    });

    it('handles logout all sessions (Phase 29.4)', () => {
        window.confirm = vi.fn().mockReturnValue(true);
        render(<SessionsSettingsSection role={UserRoles.USER} />);

        const logoutAllBtn = screen.getByText('Завершити всі сесії');
        fireEvent.click(logoutAllBtn);

        expect(mockLogoutAllSessions).toHaveBeenCalled();
    });

    it('renders loading state', () => {
        mockUseSessionsQuery.mockReturnValue({
            data: [],
            isLoading: true,
            error: null,
        });

        render(<SessionsSettingsSection role={UserRoles.USER} />);

        expect(screen.getByText('Завантаження сесій...')).toBeDefined();
    });

    it('renders error state', () => {
        mockUseSessionsQuery.mockReturnValue({
            data: [],
            isLoading: false,
            error: new Error('Failed to load sessions'),
        });

        render(<SessionsSettingsSection role={UserRoles.USER} />);

        expect(screen.getByText('Не вдалося завантажити активні сесії.')).toBeDefined();
    });

    it('uses lastSeenAt if present, fallback to createdAt (Phase 29.1)', () => {
        const sessionsWithLastSeen = [
            {
                id: 'sess-1',
                device: 'Mac',
                ipAddress: '127.0.0.1',
                createdAt: '2026-06-05T00:00:00Z',
                lastSeenAt: '2026-06-06T12:34:56Z',
                userAgent: 'macintosh',
                currentSession: true,
            },
        ];
        mockUseSessionsQuery.mockReturnValue({
            data: sessionsWithLastSeen,
            isLoading: false,
            error: null,
        });

        render(<SessionsSettingsSection role={UserRoles.USER} />);

        const expectedDateStr = new Date('2026-06-06T12:34:56Z').toLocaleString('uk-UA');
        expect(screen.getByText(new RegExp(expectedDateStr))).toBeDefined();
    });

    it('renders trusted, suspicious, and untrusted badges and approximate location (Phase 29.2)', () => {
        const specialSessions = [
            {
                id: 'sess-1',
                device: 'Mac',
                ipAddress: '127.0.0.1',
                approximateLocation: 'Kyiv, Ukraine',
                createdAt: '2026-06-05T00:00:00Z',
                trustedAt: '2026-06-05T01:00:00Z',
                userAgent: 'macintosh',
                currentSession: true,
            },
            {
                id: 'sess-2',
                device: 'iPhone',
                ipAddress: '192.168.1.1',
                approximateLocation: 'London, UK',
                createdAt: '2026-06-05T00:00:00Z',
                riskScore: 0.8,
                userAgent: 'iphone',
                currentSession: false,
            },
        ];
        mockUseSessionsQuery.mockReturnValue({
            data: specialSessions,
            isLoading: false,
            error: null,
        });

        render(<SessionsSettingsSection role={UserRoles.USER} />);

        // Verify approximate location is rendered
        expect(screen.getByText(/Kyiv, Ukraine/)).toBeDefined();
        expect(screen.getByText(/London, UK/)).toBeDefined();

        // Verify badges
        expect(screen.getByText('Довірений')).toBeDefined();
        expect(screen.getByText('Підозрілий')).toBeDefined();
        expect(screen.getByText('Невідомий')).toBeDefined();
    });
});

describe('PrivacySettingsSection', () => {
    it('renders toggles', () => {
        render(<PrivacySettingsSection />);
        expect(screen.getByText('Приховати email')).toBeDefined();
        expect(screen.getByText('Сповіщення безпеки')).toBeDefined();
    });
});

describe('DangerZoneSection', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('triggers deactivation mutation on deactivation click', () => {
        window.confirm = vi.fn().mockReturnValue(true);
        window.alert = vi.fn();
        render(<DangerZoneSection role={UserRoles.USER} />);

        const deactivateBtn = screen.getByText('Деактивувати');
        fireEvent.click(deactivateBtn);

        expect(mockDeactivateAccount).toHaveBeenCalledWith('mocked-token', expect.any(Object));
    });

    it('triggers deletion mutation on delete click', () => {
        window.confirm = vi.fn().mockReturnValue(true);
        window.alert = vi.fn();
        render(<DangerZoneSection role={UserRoles.USER} />);

        const deleteBtn = screen.getByRole('button', { name: 'Видалити назавжди' });
        fireEvent.click(deleteBtn);

        expect(mockScheduleAccountDeletion).toHaveBeenCalledWith('mocked-token', expect.any(Object));
    });
});
