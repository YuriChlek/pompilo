import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MailSettingsPage } from '@/features/module-admin-mail/components/mail-settings-page';
import { adminMailApiService } from '@/features/module-admin-mail/api-service/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import * as storage from '@/features/module-admin-mail/lib/mail-template-preview-storage';

vi.mock('@/features/module-admin-mail/api-service/client', () => ({
    adminMailApiService: {
        getSettings: vi.fn(),
        getSetupState: vi.fn(),
        sendTestEmail: vi.fn(),
        getTemplates: vi.fn(),
        getTemplatePreview: vi.fn(),
    },
}));

vi.mock('next/link', () => ({
    default: ({ children, href }: { children: React.ReactNode; href: string }) => (
        <a href={href}>{children}</a>
    ),
}));

describe('Test Email Tab - Template Selection Integration', () => {
    let queryClient: QueryClient;

    const mockTemplates = [
        {
            id: 'verification-code',
            name: 'Verification Code',
            description: 'Code desc',
            subject: 'Confirm email',
        },
        {
            id: 'password-reset',
            name: 'Password Reset',
            description: 'Reset desc',
            subject: 'Reset pass',
        },
    ];

    beforeEach(() => {
        vi.restoreAllMocks();
        vi.clearAllMocks();
        localStorage.clear();

        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                },
            },
        });

        vi.mocked(adminMailApiService.getSettings).mockResolvedValue({
            smtpHost: 'smtp.test.com',
            smtpPort: 587,
            smtpSecure: false,
            smtpUser: null,
            smtpPasswordEncrypted: null,
            fromAddress: 'noreply@test.com',
            fromName: 'Test',
            replyTo: null,
            clientPublicUrl: null,
            enabled: true,
        });

        vi.mocked(adminMailApiService.getSetupState).mockResolvedValue({
            state: 'configured',
        });

        vi.mocked(adminMailApiService.getTemplates).mockResolvedValue(mockTemplates);
    });

    it('should render a template select control loaded from backend templates', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const user = userEvent.setup();
        await user.click(screen.getByRole('tab', { name: 'Test Email' }));

        // Check select element exists
        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        expect(selectCombobox).toBeDefined();

        // The first template should be auto-selected by default
        expect(selectCombobox.textContent).toBe('Verification Code');
    });

    it('should auto-select the last viewed template id when available and valid', async () => {
        vi.spyOn(storage, 'getLastViewedMailTemplateId').mockReturnValue('password-reset');

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const user = userEvent.setup();
        await user.click(screen.getByRole('tab', { name: 'Test Email' }));

        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        expect(selectCombobox.textContent).toBe('Password Reset');
    });

    it('should fall back to the first template when last viewed template id is not available', async () => {
        vi.spyOn(storage, 'getLastViewedMailTemplateId').mockReturnValue('missing-template');

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const user = userEvent.setup();
        await user.click(screen.getByRole('tab', { name: 'Test Email' }));

        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        expect(selectCombobox.textContent).toBe('Verification Code');
    });

    it('should submit selected template id with the recipient email', async () => {
        vi.mocked(adminMailApiService.sendTestEmail).mockResolvedValue({ outboxId: 'outbox-1' });

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const user = userEvent.setup();
        await user.click(screen.getByRole('tab', { name: 'Test Email' }));

        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        await user.click(selectCombobox);
        await user.click(await screen.findByRole('option', { name: 'Password Reset' }));

        await user.type(
            screen.getByRole('textbox', { name: /recipient email/i }),
            'recipient@example.com',
        );
        await user.click(screen.getByRole('button', { name: 'Send Test' }));

        await waitFor(() => {
            expect(adminMailApiService.sendTestEmail).toHaveBeenCalledWith({
                to: 'recipient@example.com',
                templateId: 'password-reset',
            });
        });
    });

    it('should disable the send button when templates are empty', async () => {
        vi.mocked(adminMailApiService.getTemplates).mockResolvedValue([]);

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const user = userEvent.setup();
        await user.click(screen.getByRole('tab', { name: 'Test Email' }));

        const sendButton = screen.getByRole('button', { name: /send test/i });
        expect(sendButton.getAttribute('disabled')).not.toBeNull();
    });
});
