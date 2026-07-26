import { useQuery } from '@tanstack/react-query';
import { accountApiService } from '@/features/module-account/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

export const ACCOUNT_QUERY_KEYS = {
    sessions: (role: UserRoles) => ['account', 'sessions', role],
};

export const useSessionsQuery = (role: UserRoles) => {
    return useQuery({
        queryKey: ACCOUNT_QUERY_KEYS.sessions(role),
        queryFn: () => accountApiService.getActiveSessions(role),
    });
};
