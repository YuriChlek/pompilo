import type { Theme } from '@/shared/theme/types/theme.types';

export const isTheme = (value: unknown): value is Theme => value === 'light' || value === 'dark';
