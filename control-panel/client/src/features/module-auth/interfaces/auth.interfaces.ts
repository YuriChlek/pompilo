import type { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { RegisterData } from '@/features/module-auth/types/auth.types';

export interface User {
    id: string;
    name: string;
    email: string;
    role: UserRoles;
}

export interface CheckpointResponse {
    checkpointRequired: true;
    loginChallengeId: string;
    checkpointToken: string;
    expiresInSeconds: number;
    resendAvailableInSeconds: number;
}

export interface ReauthResponse {
    confirmationToken: string;
    expiresAt: string;
}

export interface AuthApi {
    login(login: string, password: string, role?: UserRoles): Promise<User | CheckpointResponse | null>;
    logout(role?: UserRoles): Promise<boolean>;
    register(data: RegisterData): Promise<User | null>;
    getMe(role?: UserRoles): Promise<User | null>;
    verifyCheckpoint(checkpointToken: string, code: string, role?: UserRoles): Promise<User | null>;
    resendCheckpoint(checkpointToken: string, role?: UserRoles): Promise<CheckpointResponse>;
    reauth(password: string, actionScope: string, role: UserRoles): Promise<ReauthResponse>;
    verifyEmail(token: string): Promise<boolean>;
    resendVerification(): Promise<void>;
    forgotPassword(email: string): Promise<void>;
    resetPassword(token: string, newPassword: string): Promise<void>;
}
