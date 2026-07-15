const LAST_VIEWED_TEMPLATE_ID_KEY = 'last_viewed_mail_template_id';

export const getLastViewedMailTemplateId = (): string | null => {
    if (typeof window === 'undefined') {
        return null;
    }
    return localStorage.getItem(LAST_VIEWED_TEMPLATE_ID_KEY);
};

export const setLastViewedMailTemplateId = (templateId: string): void => {
    if (typeof window === 'undefined') {
        return;
    }
    localStorage.setItem(LAST_VIEWED_TEMPLATE_ID_KEY, templateId);
};
