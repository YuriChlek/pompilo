export const normalizeDateOnly = (value: string | null | undefined): string => {
    return value ? value.slice(0, 10) : '';
};
