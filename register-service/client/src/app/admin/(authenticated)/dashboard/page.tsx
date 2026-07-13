'use client';

import { PageTitle } from '@/components/page-title/page-title';
import { Box, Container, Typography } from '@mui/material';

export default function AdminDashboardPage() {
    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            <PageTitle pageTitle="Admin Dashboard" />

            <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom>
                    System Management
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Select a section from the sidebar menu to begin configuration.
                </Typography>
            </Box>
        </Container>
    );
}
