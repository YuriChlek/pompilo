export const normalizeStr = (str: string) => {
    return str.trim().toLowerCase();
};

export const slugify = (str: string): string => {
    return normalizeStr(str)
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
};
