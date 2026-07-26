import * as React from 'react';

export interface MailTemplateSummary {
    id: string;
    name: string;
    description: string;
    subject: string;
}

export interface MailTemplatePreview extends MailTemplateSummary {
    html: string;
    text: string;
}

export interface MailTemplateRegistryItem extends MailTemplateSummary {
    component: React.ComponentType<any>;
    demoProps: Record<string, any>;
}
