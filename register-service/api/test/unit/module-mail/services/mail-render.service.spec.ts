import { Test, TestingModule } from '@nestjs/testing';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import * as React from 'react';

describe('MailRenderService', () => {
    let service: MailRenderService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [MailRenderService],
        }).compile();

        service = module.get<MailRenderService>(MailRenderService);
    });

    it('should render a simple component to html', async () => {
        const component = React.createElement('div', {}, 'Hello World');
        const html = await service.renderHtml(component);
        expect(html).toContain('Hello World');
        expect(html).toContain('<!DOCTYPE html');
    });

    it('should render a simple component to text', async () => {
        const component = React.createElement('div', {}, 'Hello World');
        const text = await service.renderText(component);
        expect(text).toBe('Hello World');
    });
});
