import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MailSettingsPage } from '@/features/module-admin-mail/components/mail-settings-page';
import { adminMailApiService } from '@/features/module-admin-mail/api-service/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import * as storage from '@/features/module-admin-mail/lib/mail-template-preview-storage';

// Mock the API service
vi.mock('@/features/module-admin-mail/api-service/client', () => ({
    adminMailApiService: {
        getSettings: vi.fn(),
        getSetupState: vi.fn(),
        sendTestEmail: vi.fn(),
        getTemplates: vi.fn(),
        getTemplatePreview: vi.fn(),
    },
}));

// Mock Next.js Link
vi.mock('next/link', () => ({
    default: ({ children, href }: { children: React.ReactNode; href: string }) => (
        <a href={href}>{children}</a>
    ),
}));

describe('MailTemplatePreview Integration', () => {
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

    const mockPreviewData = {
        id: 'verification-code',
        name: 'Verification Code',
        description: 'Code desc',
        subject: 'Confirm email',
        html: '<html>Verification Code HTML</html>',
        text: 'Verification Code Text',
    };

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

        // Default setup responses
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
        vi.mocked(adminMailApiService.getTemplatePreview).mockResolvedValue(mockPreviewData);
    });

    it('should render the Email Templates tab and switch to it', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        const tab = screen.getByRole('tab', { name: 'Email Templates' });
        expect(tab).toBeDefined();

        fireEvent.click(tab);

        await waitFor(() => {
            expect(screen.getByText('Email Templates Preview')).toBeDefined();
        });
    });

    it('should load template list and auto-preview the first template by default', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(adminMailApiService.getTemplates).toHaveBeenCalled();
            expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledWith(
                'verification-code',
            );
        });

        await waitFor(() => {
            expect(screen.getByText('Confirm email')).toBeDefined();
        });
        expect(screen.queryByText('Code desc')).toBeNull();

        // Verify HTML frame and plain text
        const iframe = screen.getByTitle('Email HTML Preview') as HTMLIFrameElement;
        expect(iframe).toBeDefined();
        expect(iframe.getAttribute('srcdoc')).toBe('<html>Verification Code HTML</html>');

        // Switch to Plain Text tab to verify plain text content
        fireEvent.click(screen.getByRole('tab', { name: 'Plain Text' }));
        expect(screen.getByText('Verification Code Text')).toBeDefined();
    });

    it('should auto-preview the last viewed template if it is in the list', async () => {
        vi.spyOn(storage, 'getLastViewedMailTemplateId').mockReturnValue('password-reset');
        const previewReset = {
            ...mockPreviewData,
            id: 'password-reset',
            name: 'Password Reset',
            subject: 'Reset pass',
            html: '<html>Password Reset HTML</html>',
            text: 'Password Reset Text',
        };
        vi.mocked(adminMailApiService.getTemplatePreview).mockResolvedValue(previewReset);

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledWith('password-reset');
        });

        await waitFor(() => {
            expect(screen.getByText('Reset pass')).toBeDefined();
        });
        const iframe = screen.getByTitle('Email HTML Preview') as HTMLIFrameElement;
        expect(iframe.getAttribute('srcdoc')).toBe('<html>Password Reset HTML</html>');
    });

    it('should NOT update preview when changing select value until View Template is clicked', async () => {
        const user = userEvent.setup();

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledWith(
                'verification-code',
            );
        });

        // Change select value to password-reset
        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        fireEvent.mouseDown(selectCombobox);
        const option = await screen.findByRole('option', { name: 'Password Reset' });
        fireEvent.click(option);

        // Verification: getTemplatePreview was NOT called again for password-reset
        expect(adminMailApiService.getTemplatePreview).not.toHaveBeenCalledWith('password-reset');

        // Click View Template
        const previewReset = {
            ...mockPreviewData,
            id: 'password-reset',
            name: 'Password Reset',
            subject: 'Reset pass',
            html: '<html>Password Reset HTML</html>',
            text: 'Password Reset Text',
        };
        vi.mocked(adminMailApiService.getTemplatePreview).mockResolvedValue(previewReset);

        const button = screen.getByRole('button', { name: 'View Template' });
        await user.click(button);

        await waitFor(() => {
            expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledWith('password-reset');
        });

        await waitFor(() => {
            expect(screen.getByText('Reset pass')).toBeDefined();
        });
    });

    it('should save the template ID to storage after a successful preview click', async () => {
        const user = userEvent.setup();
        const setStorageSpy = vi.spyOn(storage, 'setLastViewedMailTemplateId');

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledWith(
                'verification-code',
            );
        });

        // Change select to password-reset and click View Template
        const selectCombobox = screen.getByRole('combobox', { name: /template/i });
        fireEvent.mouseDown(selectCombobox);
        const option = await screen.findByRole('option', { name: 'Password Reset' });
        fireEvent.click(option);

        const button = screen.getByRole('button', { name: 'View Template' });
        await user.click(button);

        await waitFor(() => {
            expect(setStorageSpy).toHaveBeenCalledWith('password-reset');
        });
    });

    it('should reuse cached preview after switching away from and back to the Email Templates tab', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(screen.getByText('Confirm email')).toBeDefined();
        });

        expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledTimes(1);

        fireEvent.click(screen.getByRole('tab', { name: 'General Settings' }));
        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'View Template' })).toBeDefined();
        });

        const select = screen.getByRole('combobox', { name: /template/i });
        expect(select.getAttribute('aria-disabled')).not.toBe('true');
        expect(screen.queryByRole('button', { name: /loading preview/i })).toBeNull();
        expect(adminMailApiService.getTemplatePreview).toHaveBeenCalledTimes(1);
    });

    it('should disable inputs/buttons and show empty state when templates list is empty', async () => {
        vi.mocked(adminMailApiService.getTemplates).mockResolvedValue([]);

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        fireEvent.click(screen.getByRole('tab', { name: 'Email Templates' }));

        await waitFor(() => {
            expect(screen.getByText('No email templates available on the server.')).toBeDefined();
        });

        const select = screen.getByRole('combobox', { name: /template/i });
        const button = screen.getByRole('button', { name: 'View Template' });

        expect(select.getAttribute('aria-disabled')).toBe('true');
        expect(button.hasAttribute('disabled')).toBe(true);
        expect(adminMailApiService.getTemplatePreview).not.toHaveBeenCalled();
    });
});
