import { UserRoles } from '@/module-auth/enums/auth.enums';
import { UserModel } from '@/module-user/types/user.types';

const userSequence = 1;

export const buildUserEntity = (overrides: Partial<UserModel> = {}): UserModel => ({
    id: overrides.id ?? `user-${userSequence}`,
    name: overrides.name ?? `User ${userSequence}`,
    email: overrides.email ?? `fixture${userSequence}@example.com`,
    password: overrides.password ?? 'hashed',
    role: overrides.role ?? UserRoles.USER,
    isActive: overrides.isActive ?? true,
    emailVerifiedAt: overrides.emailVerifiedAt ?? null,
    pendingEmailChange: overrides.pendingEmailChange ?? null,
    deletionScheduledAt: overrides.deletionScheduledAt ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
    tokens: overrides.tokens ?? [],
});
