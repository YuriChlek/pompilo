'use client';

import { createTheme, ThemeProvider } from '@mui/material/styles';

const ADMIN_CONTROL_HEIGHT = 40;

const adminTheme = createTheme({
    components: {
        MuiTextField: {
            defaultProps: {
                size: 'small',
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    maxHeight: 36,
                },
            },
        },
        MuiSelect: {
            defaultProps: {
                size: 'small',
            },
            styleOverrides: {
                select: {
                    minHeight: '0 !important',
                    display: 'flex',
                    alignItems: 'center',
                },
            },
        },
        MuiInputBase: {
            styleOverrides: {
                root: {
                    '&:not(.MuiInputBase-multiline)': {
                        height: ADMIN_CONTROL_HEIGHT,
                        minHeight: ADMIN_CONTROL_HEIGHT,
                    },
                },
            },
        },
    },
});

export function AdminMuiThemeProvider({ children }: Readonly<{ children: React.ReactNode }>) {
    return <ThemeProvider theme={adminTheme}>{children}</ThemeProvider>;
}
