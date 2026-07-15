import {
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
    UnauthorizedException,
} from '@nestjs/common';
import { UserDeleteResult } from '@/module-user/interfaces/user.interfaces';
import { UserModel } from '@/module-user/types/user.types';
import { UserService } from '@/module-user/services/user.service';
import { buildUserEntity } from '../../fixtures/users.fixtures';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { UserUniquenessService } from '@/module-user/services/user-uniqueness.service';
import { UserRepository } from '@/module-user/repository/user.repository';

describe('UserService', () => {
    let service: UserService;
    let repository: jest.Mocked<
        Pick<
            UserRepository,
            'save' | 'findByNameOrEmail' | 'findById' | 'findByLogin' | 'update' | 'delete'
        >
    >;
    let userPasswordService: {
        hashPassword: jest.MockedFunction<UserPasswordService['hashPassword']>;
    };
    let userUniquenessService: {
        ensureUnique: jest.MockedFunction<UserUniquenessService['ensureUnique']>;
    };

    const baseUser: UserModel = buildUserEntity({
        id: 'user-id',
        name: 'john',
        email: 'john@example.com',
    });

    beforeEach(() => {
        repository = {
            save: jest.fn(),
            findByNameOrEmail: jest.fn(),
            findById: jest.fn(),
            findByLogin: jest.fn(),
            update: jest.fn(),
            delete: jest.fn(),
        };
        userPasswordService = {
            hashPassword: jest.fn().mockResolvedValue('hashed'),
        };
        userUniquenessService = {
            ensureUnique: jest.fn(),
        };
        repository.findByNameOrEmail.mockResolvedValue([]);
        repository.save.mockResolvedValue(baseUser);
        repository.update.mockResolvedValue(undefined);
        repository.delete.mockResolvedValue({ affected: 1, raw: {} } as UserDeleteResult);

        service = new UserService(
            repository as unknown as UserRepository,
            userPasswordService as unknown as UserPasswordService,
            userUniquenessService as unknown as UserUniquenessService,
        );
        jest.clearAllMocks();
    });

    describe('create', () => {
        const dto: Parameters<UserService['create']>[0] = {
            name: 'john',
            email: 'john@example.com',
            password: 'Secret123',
        };

        it('hashes password and saves user when unique', async () => {
            const result = await service.create(dto);

            expect(repository.save).toHaveBeenCalledWith(
                expect.objectContaining({ password: 'hashed' }),
                undefined,
            );
            expect(result).toBe(baseUser);
        });

        it('throws ConflictException when user already exists', async () => {
            userUniquenessService.ensureUnique.mockRejectedValue(
                new ConflictException('duplicate'),
            );

            await expect(service.create(dto)).rejects.toBeInstanceOf(ConflictException);
        });

        it('converts repository unique violations into ConflictException', async () => {
            const uniqueError = Object.assign(new Error('duplicate'), { code: '23505' });
            repository.save.mockRejectedValue(uniqueError);

            await expect(service.create(dto)).rejects.toBeInstanceOf(ConflictException);
        });
    });

    describe('findAll', () => {
        it('returns repository results', async () => {
            repository.findByNameOrEmail.mockResolvedValue([baseUser]);

            const result = await service.findAll({
                name: 'john',
                email: 'john@example.com',
                password: 'Secret123',
            });

            expect(result).toEqual([baseUser]);
        });
    });

    describe('findById', () => {
        it('returns stored user by id', async () => {
            repository.findById.mockResolvedValue(baseUser);

            const result = await service.findById('user-id');

            expect(result).toBe(baseUser);
        });
    });

    describe('findByLogin', () => {
        it('returns user when login matches email or username', async () => {
            repository.findByLogin.mockResolvedValue(baseUser);

            const result = await service.findByLogin('john');

            expect(result).toBe(baseUser);
        });

        it('throws UnauthorizedException when user is missing', async () => {
            repository.findByLogin.mockResolvedValue(null);

            await expect(service.findByLogin('missing')).rejects.toBeInstanceOf(
                UnauthorizedException,
            );
        });
    });

    describe('update', () => {
        const updateDto: Parameters<UserService['update']>[1] = {
            name: 'New Name',
            password: 'NewPass123',
        };

        it('updates existing user and returns latest entity', async () => {
            repository.findById
                .mockResolvedValueOnce(baseUser)
                .mockResolvedValueOnce({ ...baseUser, name: 'New Name' } as UserModel);

            const result = await service.update('user-id', updateDto);

            expect(repository.update).toHaveBeenCalledWith(
                'user-id',
                expect.objectContaining({ name: 'New Name', password: 'hashed' }),
            );
            expect(result).toMatchObject({ name: 'New Name' });
        });

        it('throws InternalServerErrorException when target user does not exist', async () => {
            repository.findById.mockResolvedValueOnce(null);

            await expect(service.update('missing', updateDto)).rejects.toBeInstanceOf(
                NotFoundException,
            );
            expect(repository.update).not.toHaveBeenCalled();
        });

        it('throws ConflictException when duplicate user found', async () => {
            repository.findById.mockResolvedValue(baseUser);
            userUniquenessService.ensureUnique.mockRejectedValue(
                new ConflictException('duplicate'),
            );

            await expect(service.update('user-id', updateDto)).rejects.toBeInstanceOf(
                ConflictException,
            );
        });

        it('converts update unique violations into ConflictException', async () => {
            repository.findById.mockResolvedValue(baseUser);
            const uniqueError = Object.assign(new Error('duplicate'), { code: '23505' });
            repository.update.mockRejectedValue(uniqueError);

            await expect(service.update('user-id', updateDto)).rejects.toBeInstanceOf(
                ConflictException,
            );
        });

        it('wraps unexpected repository errors into InternalServerErrorException', async () => {
            repository.findById.mockResolvedValue(baseUser);
            repository.update.mockRejectedValue(new Error('db down'));

            await expect(service.update('user-id', updateDto)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });
    });

    describe('remove', () => {
        it('deletes an existing user', async () => {
            repository.findById.mockResolvedValue(baseUser);

            await service.remove('user-id');

            expect(repository.delete).toHaveBeenCalledWith('user-id');
        });

        it('throws NotFoundException when user is missing', async () => {
            repository.findById.mockResolvedValue(null);

            await expect(service.remove('user-id')).rejects.toBeInstanceOf(NotFoundException);
        });

        it('wraps unexpected delete errors into InternalServerErrorException', async () => {
            repository.findById.mockResolvedValue(baseUser);
            repository.delete.mockRejectedValue(new Error('db error'));

            await expect(service.remove('user-id')).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });
    });
});
