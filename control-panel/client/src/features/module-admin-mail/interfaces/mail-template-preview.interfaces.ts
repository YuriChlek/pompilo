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
