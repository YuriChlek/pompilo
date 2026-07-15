'use server';

import { cookies } from 'next/headers';
import { THEME_COOKIE, THEME_COOKIE_MAX_AGE } from '@/shared/theme/config/theme.config';
import { isTheme } from '@/shared/theme/lib/is-theme';
import type { Theme } from '@/shared/theme/types/theme.types';

export async function persistTheme(theme: Theme) {
    if (!isTheme(theme)) {
        throw new Error('Unsupported theme value');
    }
    const cookieStore = await cookies();
    cookieStore.set({
        name: THEME_COOKIE,
        value: theme,
        maxAge: THEME_COOKIE_MAX_AGE,
        path: '/',
        httpOnly: false,
        sameSite: 'lax',
    });
}
