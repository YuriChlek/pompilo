import { Injectable } from '@nestjs/common';
import { render } from '@react-email/render';
import * as React from 'react';

@Injectable()
export class MailRenderService {
    async renderHtml(component: React.ReactElement): Promise<string> {
        return await render(component);
    }

    async renderText(component: React.ReactElement): Promise<string> {
        return await render(component, { plainText: true });
    }
}
