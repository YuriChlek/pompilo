import { Body, Controller, INestApplication, Post } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import { APP_GUARD } from '@nestjs/core';
import request from 'supertest';
import { EmailFlowRateLimit } from '@/common/rate-limiting/decorators/email-flow-rate-limit.decorator';
import { EmailFlowRateLimitGuard } from '@/common/rate-limiting/guards/email-flow-rate-limit.guard';
import { EmailFlowRateLimitService } from '@/common/rate-limiting/services/email-flow-rate-limit.service';
import { EMAIL_FLOW_RATE_LIMIT_MESSAGE } from '@/common/rate-limiting/constants/email-flow-rate-limit.constants';

const dbLookup = jest.fn();

@Controller()
class TestEmailFlowController {
    @Post('register')
    @EmailFlowRateLimit({ flow: 'registration', recipientBodyField: 'email' })
    register(@Body() body: { email?: string }) {
        dbLookup(body.email);
        return { success: true };
    }

    @Post('open')
    open() {
        dbLookup('open');
        return { success: true };
    }
}

describe('EmailFlowRateLimitGuard', () => {
    let app: INestApplication;
    let rateLimitService: { check: jest.Mock };

    beforeEach(async () => {
        dbLookup.mockReset();
        rateLimitService = {
            check: jest.fn().mockResolvedValue({ limited: false, retryAfterSeconds: 0 }),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [TestEmailFlowController],
            providers: [
                { provide: EmailFlowRateLimitService, useValue: rateLimitService },
                {
                    provide: APP_GUARD,
                    useClass: EmailFlowRateLimitGuard,
                },
            ],
        }).compile();

        app = module.createNestApplication();
        await app.init();
    });

    afterEach(async () => {
        await app.close();
    });

    it('rejects rate-limited requests before the handler can perform DB work', async () => {
        rateLimitService.check.mockResolvedValue({ limited: true, retryAfterSeconds: 42 });

        const response = await request(app.getHttpServer() as never)
            .post('/register')
            .set('x-forwarded-for', '203.0.113.10, 10.0.0.1')
            .send({ email: 'victim@example.com' });

        expect(response.status).toBe(429);
        expect(response.headers['retry-after']).toBe('42');
        const body = response.body as { message?: string };
        expect(body.message).toBe(EMAIL_FLOW_RATE_LIMIT_MESSAGE);
        expect(dbLookup).not.toHaveBeenCalled();
        expect(rateLimitService.check).toHaveBeenCalledWith({
            flow: 'registration',
            ipAddress: '127.0.0.1',
            recipientEmail: 'victim@example.com',
        });
    });

    it('allows non-limited requests to reach the handler', async () => {
        const response = await request(app.getHttpServer() as never)
            .post('/register')
            .send({ email: 'user@example.com' });

        expect(response.status).toBe(201);
        expect(dbLookup).toHaveBeenCalledWith('user@example.com');
    });

    it('does not touch Redis for routes without email flow metadata', async () => {
        const response = await request(app.getHttpServer() as never)
            .post('/open')
            .send({});

        expect(response.status).toBe(201);
        expect(dbLookup).toHaveBeenCalledWith('open');
        expect(rateLimitService.check).not.toHaveBeenCalled();
    });
});
