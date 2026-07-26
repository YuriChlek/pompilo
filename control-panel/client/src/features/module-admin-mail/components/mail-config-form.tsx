'use client';

import React, { useState } from 'react';
import {
    Card,
    CardContent,
    Typography,
    TextField,
    Button,
    Box,
    FormControlLabel,
    Switch,
    Grid,
    CircularProgress,
} from '@mui/material';
import { MailSettings, UpdateMailSettingsDto } from '../interfaces/admin-mail.interfaces';
import { adminMailButtonSx, adminMailTextFieldSx } from '../constants/admin-mail-ui.constants';

interface MailConfigFormProps {
    settings?: MailSettings;
    onSave: (data: UpdateMailSettingsDto) => void;
    isSaving?: boolean;
}

export const MailConfigForm = ({ settings, onSave, isSaving }: MailConfigFormProps) => {
    const [formData, setFormData] = useState<UpdateMailSettingsDto>(() => ({
        smtpHost: settings?.smtpHost || '',
        smtpPort: settings?.smtpPort || 1025,
        smtpSecure: settings?.smtpSecure || false,
        smtpUser: settings?.smtpUser || '',
        fromName: settings?.fromName || '',
        fromAddress: settings?.fromAddress || '',
        replyTo: settings?.replyTo || '',
        clientPublicUrl: settings?.clientPublicUrl || '',
        enabled: settings?.enabled ?? true,
    }));

    const handleChange =
        (field: keyof UpdateMailSettingsDto) =>
        (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
            const target = e.target as HTMLInputElement;
            const value = target.type === 'checkbox' ? target.checked : target.value;
            setFormData(prev => ({ ...prev, [field]: value }));
        };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSave(formData);
    };

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    SMTP Configuration
                </Typography>
                <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={8}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="SMTP Host"
                                value={formData.smtpHost || ''}
                                onChange={handleChange('smtpHost')}
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="SMTP Port"
                                type="number"
                                value={formData.smtpPort || ''}
                                onChange={e =>
                                    setFormData(prev => ({
                                        ...prev,
                                        smtpPort: parseInt(e.target.value, 10),
                                    }))
                                }
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={formData.smtpSecure || false}
                                        onChange={handleChange('smtpSecure')}
                                        disabled={isSaving}
                                    />
                                }
                                label="Secure (TLS/SSL)"
                            />
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={formData.enabled || false}
                                        onChange={handleChange('enabled')}
                                        disabled={isSaving}
                                    />
                                }
                                label="System Enabled"
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="SMTP User"
                                value={formData.smtpUser || ''}
                                onChange={handleChange('smtpUser')}
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="SMTP Password"
                                type="password"
                                placeholder="********"
                                onChange={handleChange('smtpPassword')}
                                disabled={isSaving}
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Typography variant="subtitle1" sx={{ mt: 2 }}>
                                Email Identity
                            </Typography>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="From Name"
                                value={formData.fromName || ''}
                                onChange={handleChange('fromName')}
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="From Address"
                                value={formData.fromAddress || ''}
                                onChange={handleChange('fromAddress')}
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="Reply-To Address"
                                value={formData.replyTo || ''}
                                onChange={handleChange('replyTo')}
                                disabled={isSaving}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                size="small"
                                sx={adminMailTextFieldSx}
                                label="Client Public URL"
                                value={formData.clientPublicUrl || ''}
                                onChange={handleChange('clientPublicUrl')}
                                disabled={isSaving}
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Box
                                sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 3 }}
                            >
                                <Button
                                    type="submit"
                                    variant="contained"
                                    color="primary"
                                    size="small"
                                    sx={adminMailButtonSx}
                                    disabled={isSaving}
                                    startIcon={
                                        isSaving && <CircularProgress size={20} color="inherit" />
                                    }
                                >
                                    {isSaving ? 'Saving...' : 'Save Changes'}
                                </Button>
                            </Box>
                        </Grid>
                    </Grid>
                </Box>
            </CardContent>
        </Card>
    );
};
