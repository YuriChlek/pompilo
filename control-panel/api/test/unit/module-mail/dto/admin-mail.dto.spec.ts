import { validate } from 'class-validator';
import { SendTestEmailDto } from '@/module-mail/dto/admin-mail.dto';
import { plainToInstance } from 'class-transformer';

describe('SendTestEmailDto validation', () => {
    it('should pass with valid data', async () => {
        const dto = plainToInstance(SendTestEmailDto, {
            to: 'recipient@example.com',
            templateId: 'security-alert',
        });
        const errors = await validate(dto);
        expect(errors.length).toBe(0);
    });

    it('should fail when templateId is missing', async () => {
        const dto = plainToInstance(SendTestEmailDto, {
            to: 'recipient@example.com',
        });
        const errors = await validate(dto);
        expect(errors.length).toBeGreaterThan(0);
        expect(errors[0].property).toBe('templateId');
    });

    it('should fail with invalid recipient email', async () => {
        const dto = plainToInstance(SendTestEmailDto, {
            to: 'invalid-email',
            templateId: 'security-alert',
        });
        const errors = await validate(dto);
        expect(errors.length).toBeGreaterThan(0);
        expect(errors[0].property).toBe('to');
    });
});
