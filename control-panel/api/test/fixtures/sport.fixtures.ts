import { SportModel } from '@/module-sport/types/sport.types';

export function buildSportEntity(overrides: Partial<SportModel> = {}): SportModel {
    return {
        id: 'sport-id',
        name: 'Fitness',
        slug: 'fitness',
        createdAt: new Date(),
        updatedAt: new Date(),
        ...overrides,
    };
}
