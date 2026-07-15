import { UserRoles } from '@/module-auth/enums/auth.enums';
import { LoginUserDto } from '@/module-auth/dto/login-user.dto';
import { RegisterUserDto } from '@/module-auth/dto/register-user.dto';

let sequence = 1;

export const buildRegisterDto = (overrides: Partial<RegisterUserDto> = {}) =>
    ({
        name: `Test User ${sequence}`,
        email: `user${sequence++}@example.com`,
        password: 'Password1',
        role: UserRoles.USER,
        ...overrides,
    }) satisfies RegisterUserDto;

export const buildLoginDto = (overrides: Partial<LoginUserDto> = {}) =>
    ({
        login: overrides.login ?? `user${sequence}@example.com`,
        password: overrides.password ?? 'Password1',
    }) satisfies LoginUserDto;
