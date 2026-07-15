import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Payment Settings | Admin Panel',
};

export default function Page() {
    return (
        <section style={{ padding: '32px' }}>
            <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 700 }}>Payment Settings</h1>
            <p style={{ marginTop: '12px', color: '#6b7280' }}>
                Payment method settings are not configured yet.
            </p>
        </section>
    );
}
