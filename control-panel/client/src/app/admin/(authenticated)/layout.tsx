import { AdminMuiThemeProvider } from './admin-mui-theme-provider';
import { AdminSidebar } from '@/features/module-admin-shell';
import styles from './layout.module.css';

export default function AdminLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <AdminMuiThemeProvider>
            <div className={styles.container}>
                <AdminSidebar />
                <main className={`${styles.mainContent} page`}>{children}</main>
            </div>
        </AdminMuiThemeProvider>
    );
}
