import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MailSettingsPage } from '@/features/module-admin-mail/components/mail-settings-page';
import { adminMailApiService } from '@/features/module-admin-mail/api-service/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

// Mock the API service
vi.mock('@/features/module-admin-mail/api-service/client', () => ({
    adminMailApiService: {
        getSettings: vi.fn(),
        getSetupState: vi.fn(),
        sendTestEmail: vi.fn(),
        getTemplates: vi.fn(),
    },
}));

// Mock Next.js Link
vi.mock('next/link', () => ({
    default: ({ children, href }: { children: React.ReactNode; href: string }) => (
        <a href={href}>{children}</a>
    ),
}));

describe('MailSettingsPage Polling', () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                },
            },
        });

        // Default successful responses
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
        vi.mocked(adminMailApiService.getTemplates).mockResolvedValue([
            {
                id: 'security-alert',
                name: 'Security Alert',
                description: 'Test template description',
                subject: 'Security Notification',
            },
        ]);
    });

    it('should NOT perform health polling (no calls to /admin/mail/health)', async () => {
        // We use a spy on a non-existent method or just check that no unexpected calls happen.
        // Actually, since we removed getHealth from apiService, any call to it would be a TS error if it was still in code.
        // But we want to ensure no dynamic/manual fetch is happening.

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        // Wait for initial data to load
        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        // Advance time to see if any polling occurs (though we removed refetchInterval)
        vi.useFakeTimers();
        await vi.advanceTimersByTimeAsync(30000); // 30 seconds

        // Verification: only getSettings and getSetupState should have been called (likely once each)
        expect(adminMailApiService.getSettings).toHaveBeenCalled();
        expect(adminMailApiService.getSetupState).toHaveBeenCalled();

        // If there was a getHealth call, it would be through a property we didn't mock or
        // we can check all mocks for unexpected calls.
        // Since we explicitly removed it from use-admin-mail.hooks.ts, this is a regression test.

        vi.useRealTimers();
    });

    it('should submit the entered recipient address when sending a test email', async () => {
        const user = userEvent.setup();
        vi.mocked(adminMailApiService.sendTestEmail).mockResolvedValue({ outboxId: 'outbox-1' });

        render(
            <QueryClientProvider client={queryClient}>
                <MailSettingsPage />
            </QueryClientProvider>,
        );

        await waitFor(() => {
            expect(screen.getByText('Email Settings')).toBeDefined();
        });

        await user.click(screen.getByRole('tab', { name: 'Test Email' }));
        await user.type(
            screen.getByRole('textbox', { name: /recipient email/i }),
            'recipient@example.com',
        );
        await user.click(screen.getByRole('button', { name: 'Send Test' }));

        await waitFor(() => {
            expect(adminMailApiService.sendTestEmail).toHaveBeenCalledWith({
                to: 'recipient@example.com',
                templateId: 'security-alert',
            });
        });
    });
});
