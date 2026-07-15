import { Controller, Get, type Provider } from '@nestjs/common';
import { MODULE_METADATA } from '@nestjs/common/constants';
import { Reflector } from '@nestjs/core';
import { Test, TestingModule } from '@nestjs/testing';
import { AuthModule } from '@/module-auth/auth.module';
import { Authorisation, ROLES_KEY } from '@/module-auth/decorators/auth.decorator';
import { JwtAuthGuard } from '@/module-auth/guards/jwt-auth.guard';
import { RolesGuard } from '@/module-auth/guards/roles.guard';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';

@Controller('protected-test')
class ProtectedTestController {
    @Get()
    @Authorisation(UserRoles.USER)
    getProtected(this: void) {
        return { ok: true };
    }
}

describe('Auth guards DI wiring', () => {
    let moduleRef: TestingModule;

    beforeAll(async () => {
        const authModuleProviders = Reflect.getMetadata(
            MODULE_METADATA.PROVIDERS,
            AuthModule,
        ) as Provider[];
        const guardProviders = authModuleProviders.filter(
            provider => provider === JwtAuthGuard || provider === RolesGuard,
        );

        moduleRef = await Test.createTestingModule({
            controllers: [ProtectedTestController],
            providers: [
                ...guardProviders,
                {
                    provide: RedisTokenService,
                    useValue: {
                        isTokenRevoked: jest.fn().mockResolvedValue(false),
                        isSessionRevoked: jest.fn().mockResolvedValue(false),
                    },
                },
            ],
        }).compile();
    });

    afterAll(async () => {
        await moduleRef?.close();
    });

    it('registers auth guards as AuthModule providers and exports', () => {
        const providers = Reflect.getMetadata(MODULE_METADATA.PROVIDERS, AuthModule) as unknown[];
        const exports = Reflect.getMetadata(MODULE_METADATA.EXPORTS, AuthModule) as unknown[];

        expect(providers).toContain(JwtAuthGuard);
        expect(providers).toContain(RolesGuard);
        expect(exports).toContain(JwtAuthGuard);
        expect(exports).toContain(RolesGuard);
    });

    it('resolves JwtAuthGuard through the Nest container', () => {
        const guard = moduleRef.get(JwtAuthGuard);

        expect(guard).toBeInstanceOf(JwtAuthGuard);
    });

    it('resolves RolesGuard through the Nest container', () => {
        const guard = moduleRef.get(RolesGuard);

        expect(guard).toBeInstanceOf(RolesGuard);
    });

    it('keeps @Authorisation role metadata on protected handlers', () => {
        const reflector = moduleRef.get(Reflector);
        const controller = moduleRef.get(ProtectedTestController);
        const roles = reflector.getAllAndOverride<UserRoles[]>(ROLES_KEY, [
            controller.getProtected,
            ProtectedTestController,
        ]);

        expect(roles).toEqual([UserRoles.USER]);
    });
});
