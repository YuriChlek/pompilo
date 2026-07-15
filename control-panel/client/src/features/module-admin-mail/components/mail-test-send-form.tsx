'use client';

import React from 'react';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    CircularProgress,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    TextField,
    Typography,
} from '@mui/material';
import {
    adminMailButtonSx,
    adminMailSelectMenuProps,
    adminMailSelectSx,
    adminMailTextFieldSx,
} from '../constants/admin-mail-ui.constants';
import { MailTemplateSummary } from '../interfaces/mail-template-preview.interfaces';
import { getLastViewedMailTemplateId } from '../lib/mail-template-preview-storage';

interface MailTestSendFormProps {
    disabled?: boolean;
    isLoading?: boolean;
    templates?: MailTemplateSummary[];
    templatesLoading?: boolean;
    onSend: (to: string, templateId: string) => void;
}

export const MailTestSendForm = ({
    disabled,
    isLoading,
    templates,
    templatesLoading,
    onSend,
}: MailTestSendFormProps) => {
    const [recipientEmail, setRecipientEmail] = React.useState('');
    const [userSelectedTemplateId, setUserSelectedTemplateId] = React.useState('');

    // Phase 7. Initial Selection Behaviour
    const initialTemplateId = React.useMemo(() => {
        if (!templates || templates.length === 0) {
            return '';
        }

        const lastViewedId = getLastViewedMailTemplateId();

        return lastViewedId && templates.some(t => t.id === lastViewedId)
            ? lastViewedId
            : templates[0].id;
    }, [templates]);
    const hasUserSelectedTemplate =
        userSelectedTemplateId && templates?.some(t => t.id === userSelectedTemplateId);
    const selectedTemplateId = hasUserSelectedTemplate
        ? userSelectedTemplateId
        : initialTemplateId;

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (selectedTemplateId) {
            onSend(recipientEmail.trim(), selectedTemplateId);
        }
    };

    const hasTemplates = templates && templates.length > 0;
    const isControlDisabled = disabled || isLoading || templatesLoading;

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Test Email
                </Typography>

                {disabled && (
                    <Alert severity="error" sx={{ mb: 3 }}>
                        Email system is not configured. Save SMTP settings before sending a test
                        email.
                    </Alert>
                )}

                {!templatesLoading && !hasTemplates && (
                    <Alert severity="warning" sx={{ mb: 3 }}>
                        No email templates available on the server.
                    </Alert>
                )}

                <Box
                    component="form"
                    onSubmit={handleSubmit}
                    sx={{
                        mt: 2,
                        display: 'grid',
                        gridTemplateColumns: 'minmax(240px, 1fr) minmax(260px, 1fr) auto',
                        gap: 3,
                        alignItems: 'flex-start',
                    }}
                >
                    <FormControl fullWidth disabled={isControlDisabled || !hasTemplates}>
                        <InputLabel id="test-email-template-select-label">Template</InputLabel>
                        <Select
                            labelId="test-email-template-select-label"
                            id="test-email-template-select"
                            size="small"
                            sx={adminMailSelectSx}
                            MenuProps={adminMailSelectMenuProps}
                            value={selectedTemplateId}
                            label="Template"
                            onChange={event => setUserSelectedTemplateId(event.target.value)}
                        >
                            {templates?.map(t => (
                                <MenuItem key={t.id} value={t.id}>
                                    {t.name}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    <TextField
                        fullWidth
                        required
                        size="small"
                        sx={adminMailTextFieldSx}
                        type="email"
                        label="Recipient email"
                        value={recipientEmail}
                        onChange={event => setRecipientEmail(event.target.value)}
                        disabled={isControlDisabled}
                    />

                    <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        size="small"
                        sx={{ ...adminMailButtonSx, minHeight: 40, whiteSpace: 'nowrap' }}
                        disabled={
                            disabled ||
                            isLoading ||
                            templatesLoading ||
                            !selectedTemplateId ||
                            !recipientEmail.trim()
                        }
                        startIcon={isLoading && <CircularProgress size={20} color="inherit" />}
                    >
                        {isLoading ? 'Sending...' : 'Send Test'}
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
};
