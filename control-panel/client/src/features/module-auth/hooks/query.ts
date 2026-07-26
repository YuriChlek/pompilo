import { useQuery } from '@tanstack/react-query';
import { authService } from '@/features/module-auth/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';

export const getUserQueryKey = (userRole?: UserRoles) =>
    userRole ? [`${userRole}Data`] : ['userData'];

export const useUser = (userRole?: UserRoles, initialData?: User | null) => {
    const queryKey = getUserQueryKey(userRole);

    return useQuery<User | null>({
        queryKey,
        queryFn: (): Promise<User | null> => authService.getMe(userRole),
        initialData,
        gcTime: 300000,
        staleTime: initialData ? 300000 : 0, // Force refetch if no initial data
        refetchOnWindowFocus: false,
    });
};
