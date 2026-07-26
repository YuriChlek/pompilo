import type { TokenInsert, TokenSelect } from '@/module-auth-token/schemas';
import type { User } from '@/module-user/interfaces/user.interfaces';

export type TokenUserPayload = User;
export type TokenModel = TokenSelect;
export type NewTokenModel = TokenInsert & Required<Pick<TokenInsert, 'id'>>;
