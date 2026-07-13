import { Test, TestingModule } from '@nestjs/testing';
import { RequestMethod } from '@nestjs/common';
import { PATH_METADATA, METHOD_METADATA } from '@nestjs/common/constants';
import { AdminMailTemplateController } from '@/module-mail/controllers/admin-mail-template.controller';
import { MailTemplatePreviewService } from '@/module-mail/services/mail-template-preview.service';
import { ROLES_KEY } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';

describe('AdminMailTemplateController', () => {
    let controller: AdminMailTemplateController;
    let previewService: {
        getTemplateSummaries: jest.Mock;
        getTemplatePreview: jest.Mock;
    };

    const getHandler = (
        methodName: keyof AdminMailTemplateController,
    ): ((...args: unknown[]) => unknown) =>
        Object.getOwnPropertyDescriptor(AdminMailTemplateController.prototype, methodName)!
            .value as (...args: unknown[]) => unknown;

    beforeEach(async () => {
        previewService = {
            getTemplateSummaries: jest.fn(),
            getTemplatePreview: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [AdminMailTemplateController],
            providers: [{ provide: MailTemplatePreviewService, useValue: previewService }],
        }).compile();

        controller = module.get<AdminMailTemplateController>(AdminMailTemplateController);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    it('should delegate getTemplates to preview service', () => {
        const summaries = [{ id: 't1', name: 'T1', description: 'D1', subject: 'S1' }];
        previewService.getTemplateSummaries.mockReturnValue(summaries);

        const result = controller.getTemplates();
        expect(previewService.getTemplateSummaries).toHaveBeenCalledTimes(1);
        expect(result).toEqual(summaries);
    });

    it('should delegate getTemplatePreview to preview service', async () => {
        const preview = {
            id: 't1',
            name: 'T1',
            description: 'D1',
            subject: 'S1',
            html: '<p>HTML</p>',
            text: 'text',
        };
        previewService.getTemplatePreview.mockResolvedValue(preview);

        const result = await controller.getTemplatePreview('t1');
        expect(previewService.getTemplatePreview).toHaveBeenCalledWith('t1');
        expect(result).toEqual(preview);
    });

    it('should have correct route metadata for class and methods', () => {
        expect(Reflect.getMetadata(PATH_METADATA, AdminMailTemplateController)).toBe(
            'admin/mail/templates',
        );

        const listHandler = getHandler('getTemplates');
        expect(Reflect.getMetadata(PATH_METADATA, listHandler)).toBe('/');
        expect(Reflect.getMetadata(METHOD_METADATA, listHandler)).toBe(RequestMethod.GET);

        const previewHandler = getHandler('getTemplatePreview');
        expect(Reflect.getMetadata(PATH_METADATA, previewHandler)).toBe(':templateId/preview');
        expect(Reflect.getMetadata(METHOD_METADATA, previewHandler)).toBe(RequestMethod.GET);
    });

    it('should restrict access to admin and superAdmin roles', () => {
        const roles = Reflect.getMetadata(ROLES_KEY, AdminMailTemplateController) as
            | UserRoles[]
            | undefined;
        expect(roles).toContain(UserRoles.PLATFORM_ADMIN);
        expect(roles).toContain(UserRoles.PLATFORM_ADMIN);
        expect(roles).toContain(UserRoles.SUPER_ADMIN);
    });
});
