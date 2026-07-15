import { UserRoles } from '@/module-auth/enums/auth.enums';
export interface UserDeleteResult {
    affected: number;
    raw: unknown;
}

export interface User {
    id: string;
    name: string;
    email: string;
    role: UserRoles;
}

export interface UserJwtPayload extends User {
    ipAddress: string;
    sessionId?: string;
    userAgent: string;
}

export interface CheckpointResponse {
    checkpointRequired: true;
    loginChallengeId: string;
    checkpointToken: string;
    expiresInSeconds: number;
    resendAvailableInSeconds: number;
}
