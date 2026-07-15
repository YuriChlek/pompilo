import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ApiKeysPage } from '@/features/module-account/components/api-keys-page';
import { CUSTOMER_API_KEYS_MENU_ITEM } from '@/features/module-menu/config/menu.config';

describe('ApiKeysPage', () => {
    it('renders API key empty state without exposing fake credentials', () => {
        render(<ApiKeysPage />);

        expect(screen.getByRole('heading', { name: CUSTOMER_API_KEYS_MENU_ITEM.title })).toBeDefined();
        expect(screen.getByRole('heading', { name: 'Active keys' })).toBeDefined();
        expect(screen.getByText('No API keys')).toBeDefined();
        expect(screen.getByRole('button', { name: 'Create key' })).toBeDisabled();
        expect(screen.queryByText(/sk_live|pk_live|secret/i)).toBeNull();
    });
});
