import { cookies } from 'next/headers';
import { THEME_COOKIE } from '@/shared/theme/config/theme.config';
import { isTheme } from '@/shared/theme/lib/is-theme';
import type { Theme } from '@/shared/theme/types/theme.types';

export const getRequestTheme = async (): Promise<Theme> => {
    const cookieStore = await cookies();
    const cookieValue = cookieStore.get(THEME_COOKIE)?.value;

    return isTheme(cookieValue) ? cookieValue : 'light';
};
