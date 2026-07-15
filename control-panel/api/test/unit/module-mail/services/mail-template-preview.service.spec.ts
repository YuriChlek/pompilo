import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { MailTemplatePreviewService } from '@/module-mail/services/mail-template-preview.service';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';

describe('MailTemplatePreviewService', () => {
    let service: MailTemplatePreviewService;
    let mailRenderService: {
        renderHtml: jest.Mock;
        renderText: jest.Mock;
    };

    beforeEach(async () => {
        mailRenderService = {
            renderHtml: jest.fn().mockResolvedValue('<html>Demo HTML</html>'),
            renderText: jest.fn().mockResolvedValue('Demo text content'),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailTemplatePreviewService,
                { provide: MailRenderService, useValue: mailRenderService },
            ],
        }).compile();

        service = module.get<MailTemplatePreviewService>(MailTemplatePreviewService);
    });

    it('should return all template summaries', () => {
        const summaries = service.getTemplateSummaries();
        expect(summaries).toHaveLength(5);
        const first = summaries[0];
        expect(typeof first.id).toBe('string');
        expect(typeof first.name).toBe('string');
        expect(typeof first.description).toBe('string');
        expect(typeof first.subject).toBe('string');
    });

    it('should render HTML and text for a valid template ID', async () => {
        const result = await service.getTemplatePreview('verification-code');
        expect(result).toBeDefined();
        expect(result.id).toBe('verification-code');
        expect(result.html).toContain('Demo HTML');
        expect(result.text).toContain('Demo text content');
        expect(mailRenderService.renderHtml).toHaveBeenCalledWith(expect.any(Object));
        expect(mailRenderService.renderText).toHaveBeenCalledWith(expect.any(Object));
    });

    it('should throw NotFoundException for an unknown template ID', async () => {
        await expect(service.getTemplatePreview('unknown-template-id')).rejects.toThrow(
            NotFoundException,
        );
    });

    it('should not have dependencies on MailTemplateService or MAIL_SERVICE (regression check)', () => {
        const paramTypes = Reflect.getMetadata('design:paramtypes', MailTemplatePreviewService) as
            | unknown[]
            | undefined;
        if (paramTypes) {
            for (const param of paramTypes) {
                expect(param).not.toBe(MailTemplateService);
                // Also check that it doesn't match the symbol used for MAIL_SERVICE
                const paramName =
                    typeof param === 'symbol'
                        ? param.toString()
                        : (param as { name?: string }).name || '';
                expect(paramName).not.toContain('MAIL_SERVICE');
            }
        }
    });
});
