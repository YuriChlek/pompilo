'use client';

import { useState, useEffect } from 'react';
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
    Stack,
    Tab,
    Tabs,
    Typography,
} from '@mui/material';
import { useMailTemplates, useMailTemplatePreview } from '../hooks/use-admin-mail.hooks';
import {
    getLastViewedMailTemplateId,
    setLastViewedMailTemplateId,
} from '../lib/mail-template-preview-storage';
import {
    adminMailButtonSx,
    adminMailSelectMenuProps,
    adminMailSelectSx,
} from '../constants/admin-mail-ui.constants';

export const MailTemplatePreview = () => {
    const {
        data: templates,
        isLoading: isTemplatesLoading,
        error: templatesError,
    } = useMailTemplates();

    const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
    const [previewedTemplateId, setPreviewedTemplateId] = useState<string | null>(null);
    const [previewTab, setPreviewTab] = useState<number>(0);
    const {
        data: preview,
        isFetching: isPreviewLoading,
        error: previewError,
    } = useMailTemplatePreview(previewedTemplateId);

    useEffect(() => {
        if (templates && templates.length > 0 && !selectedTemplateId && !previewedTemplateId) {
            const lastViewedId = getLastViewedMailTemplateId();
            const initialId =
                lastViewedId && templates.some(t => t.id === lastViewedId)
                    ? lastViewedId
                    : templates[0].id;

            setTimeout(() => {
                setSelectedTemplateId(initialId);
                setPreviewedTemplateId(initialId);
            }, 0);
        }
    }, [templates, selectedTemplateId, previewedTemplateId]);

    useEffect(() => {
        if (preview && previewedTemplateId) {
            setLastViewedMailTemplateId(previewedTemplateId);
        }
    }, [preview, previewedTemplateId]);

    const handleViewTemplate = () => {
        if (!selectedTemplateId) return;

        setPreviewedTemplateId(selectedTemplateId);
    };

    if (isTemplatesLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (templatesError) {
        return (
            <Alert severity="error" sx={{ mb: 3 }}>
                Failed to load email templates: {templatesError.message}
            </Alert>
        );
    }

    const isEmpty = !templates || templates.length === 0;

    return (
        <Card>
            <CardContent>
                <Stack spacing={3}>
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Email Templates Preview
                        </Typography>

                        {isEmpty && (
                            <Alert severity="info" sx={{ mb: 2 }}>
                                No email templates available on the server.
                            </Alert>
                        )}

                        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mt: 2 }}>
                            <FormControl sx={{ minWidth: 240 }} disabled={isEmpty}>
                                <InputLabel id="email-template-select-label">Template</InputLabel>
                                <Select
                                    labelId="email-template-select-label"
                                    id="email-template-select"
                                    size="small"
                                    sx={adminMailSelectSx}
                                    MenuProps={adminMailSelectMenuProps}
                                    value={selectedTemplateId}
                                    label="Template"
                                    onChange={e => setSelectedTemplateId(e.target.value)}
                                >
                                    {templates?.map(t => (
                                        <MenuItem key={t.id} value={t.id}>
                                            {t.name}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>

                            <Button
                                variant="contained"
                                color="primary"
                                size="small"
                                sx={adminMailButtonSx}
                                onClick={handleViewTemplate}
                                disabled={isEmpty || isPreviewLoading || !selectedTemplateId}
                                startIcon={
                                    isPreviewLoading && (
                                        <CircularProgress size={20} color="inherit" />
                                    )
                                }
                            >
                                {isPreviewLoading ? 'Loading Preview...' : 'View Template'}
                            </Button>
                        </Box>
                    </Box>

                    {previewError && (
                        <Alert severity="error">
                            Failed to render template preview: {previewError.message}
                        </Alert>
                    )}

                    {preview && (
                        <Box>
                            <Box sx={{ mb: 3, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                                <Typography
                                    variant="subtitle2"
                                    component="span"
                                    sx={{ fontWeight: 'bold' }}
                                >
                                    Subject:{' '}
                                </Typography>
                                <Typography
                                    variant="body2"
                                    component="span"
                                    id="template-preview-subject"
                                >
                                    {preview.subject}
                                </Typography>
                            </Box>

                            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                                <Tabs
                                    value={previewTab}
                                    onChange={(_, newValue: number) => setPreviewTab(newValue)}
                                    aria-label="Template preview formats"
                                >
                                    <Tab
                                        label="HTML View"
                                        id="preview-tab-0"
                                        aria-controls="preview-panel-0"
                                    />
                                    <Tab
                                        label="Plain Text"
                                        id="preview-tab-1"
                                        aria-controls="preview-panel-1"
                                    />
                                </Tabs>
                            </Box>

                            {previewTab === 0 && (
                                <Box
                                    role="tabpanel"
                                    id="preview-panel-0"
                                    aria-labelledby="preview-tab-0"
                                >
                                    <iframe
                                        srcDoc={preview.html}
                                        sandbox=""
                                        style={{
                                            width: '100%',
                                            height: '500px',
                                            border: '1px solid',
                                            borderColor: 'var(--color-app-border, #ddd)',
                                            borderRadius: '4px',
                                            backgroundColor: '#fff',
                                        }}
                                        title="Email HTML Preview"
                                    />
                                </Box>
                            )}

                            {previewTab === 1 && (
                                <Box
                                    role="tabpanel"
                                    id="preview-panel-1"
                                    aria-labelledby="preview-tab-1"
                                >
                                    <Box
                                        component="pre"
                                        sx={{
                                            p: 2,
                                            border: '1px solid',
                                            borderColor: 'action.focus',
                                            borderRadius: '4px',
                                            backgroundColor: 'action.hover',
                                            whiteSpace: 'pre-wrap',
                                            fontFamily: 'monospace',
                                            fontSize: '0.9rem',
                                        }}
                                    >
                                        {preview.text}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}
                </Stack>
            </CardContent>
        </Card>
    );
};
