import type { UserInsert, UserSelect } from '@/module-user/schemas';

export type UserModel = UserSelect & {
    tokens: never[];
};

export type NewUserModel = UserInsert;
export type UserUpdateModel = Partial<UserInsert>;
