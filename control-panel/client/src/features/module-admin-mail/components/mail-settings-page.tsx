'use client';

import { useState } from 'react';
import {
    Container,
    Typography,
    Box,
    Alert,
    Stack,
    CircularProgress,
    Snackbar,
    Tab,
    Tabs,
} from '@mui/material';
import { MailConfigForm } from './mail-config-form';
import { MailTestSendForm } from './mail-test-send-form';
import { MailTemplatePreview } from './mail-template-preview';
import {
    useMailSettings,
    useMailSetupState,
    useSendTestMail,
    useUpdateMailSettings,
    useMailTemplates,
} from '../hooks/use-admin-mail.hooks';
import { UpdateMailSettingsDto } from '../interfaces/admin-mail.interfaces';
import { HttpError } from '@/lib/http-client/interfaces/http-client.interfaces';

function isHttpError(error: unknown): error is HttpError {
    return (
        typeof error === 'object' &&
        error !== null &&
        'statusCode' in error &&
        'success' in error &&
        (error as Record<string, unknown>).success === false
    );
}

const getErrorMessage = (err: unknown): string => {
    if (err instanceof Error) {
        return err.message;
    }
    if (isHttpError(err)) {
        if (typeof err.message === 'string') {
            return err.message;
        }
        if (typeof err.error === 'string') {
            return err.error;
        }
        if (Array.isArray(err.error)) {
            return err.error.join(', ');
        }
    }
    return 'Unknown error occurred';
};

export const MailSettingsPage = () => {
    const [activeTab, setActiveTab] = useState(0);
    const [snackbar, setSnackbar] = useState<{
        open: boolean;
        message: string;
        severity: 'success' | 'error';
    }>({
        open: false,
        message: '',
        severity: 'success',
    });

    const { data: settings, isLoading: isSettingsLoading } = useMailSettings();
    const { data: setupState } = useMailSetupState();
    const { data: templates, isLoading: isTemplatesLoading } = useMailTemplates();

    const { mutate: updateSettings, isPending: isSaving } = useUpdateMailSettings();
    const { mutate: sendTestMail, isPending: isSendingTest } = useSendTestMail();

    const showMessage = (message: string, severity: 'success' | 'error') => {
        setSnackbar({ open: true, message, severity });
    };

    const handleSave = (data: UpdateMailSettingsDto) => {
        updateSettings(data, {
            onSuccess: () => showMessage('Settings saved successfully', 'success'),
            onError: (err: unknown) => {
                const message = getErrorMessage(err);
                showMessage(`Failed to save settings: ${message}`, 'error');
            },
        });
    };

    if (isSettingsLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mb: 4,
                }}
            >
                <Typography variant="h4" component="h1">
                    Email Settings
                </Typography>
            </Box>

            {setupState?.state === 'unconfigured' && (
                <Alert severity="error" sx={{ mb: 4 }}>
                    Email system is not configured. Features like password reset will not work.
                </Alert>
            )}

            <Stack spacing={4}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs
                        value={activeTab}
                        onChange={(_, value: number) => setActiveTab(value)}
                        aria-label="Mail settings sections"
                    >
                        <Tab
                            label="General Settings"
                            id="mail-tab-0"
                            aria-controls="mail-panel-0"
                        />
                        <Tab label="Test Email" id="mail-tab-1" aria-controls="mail-panel-1" />
                        <Tab label="Email Templates" id="mail-tab-2" aria-controls="mail-panel-2" />
                    </Tabs>
                </Box>

                {activeTab === 0 && (
                    <Box role="tabpanel" id="mail-panel-0" aria-labelledby="mail-tab-0">
                        <MailConfigForm
                            key={settings?.smtpHost ? 'loaded' : 'empty'}
                            settings={settings || undefined}
                            onSave={handleSave}
                            isSaving={isSaving}
                        />
                    </Box>
                )}

                {activeTab === 1 && (
                    <Box role="tabpanel" id="mail-panel-1" aria-labelledby="mail-tab-1">
                        <MailTestSendForm
                            disabled={setupState?.state === 'unconfigured'}
                            isLoading={isSendingTest}
                            templates={templates}
                            templatesLoading={isTemplatesLoading}
                            onSend={(to: string, templateId: string) => {
                                sendTestMail(
                                    { to, templateId },
                                    {
                                        onSuccess: () => {
                                            showMessage(
                                                'Test email queued successfully',
                                                'success',
                                            );
                                        },
                                        onError: (err: unknown) => {
                                            const message = getErrorMessage(err);
                                            showMessage(
                                                `Failed to send test email: ${message}`,
                                                'error',
                                            );
                                        },
                                    },
                                );
                            }}
                        />
                    </Box>
                )}

                {activeTab === 2 && (
                    <Box role="tabpanel" id="mail-panel-2" aria-labelledby="mail-tab-2">
                        <MailTemplatePreview />
                    </Box>
                )}
            </Stack>

            <Snackbar
                open={snackbar.open}
                autoHideDuration={6000}
                onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            >
                <Alert
                    onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
                    severity={snackbar.severity}
                    variant="filled"
                    sx={{ width: '100%' }}
                >
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </Container>
    );
};
