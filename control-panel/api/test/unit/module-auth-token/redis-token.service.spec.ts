import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { RedisService } from '@/common/redis/redis.service';

describe('RedisTokenService', () => {
    let service: RedisTokenService;
    let mockRedis: {
        set: jest.Mock;
        get: jest.Mock;
        exists: jest.Mock;
        del: jest.Mock;
    };
    let mockRedisService: {
        getClient: jest.Mock;
        recordSuccess: jest.Mock;
        recordFailure: jest.Mock;
    };

    beforeEach(() => {
        mockRedis = {
            set: jest.fn(),
            get: jest.fn(),
            exists: jest.fn(),
            del: jest.fn(),
        };
        mockRedisService = {
            getClient: jest.fn().mockReturnValue(mockRedis),
            recordSuccess: jest.fn(),
            recordFailure: jest.fn(),
        };

        service = new RedisTokenService(mockRedisService as unknown as RedisService);
    });

    it('sets a key with expiration', async () => {
        await service.set('test-key', 'test-value', 3600);
        expect(mockRedis.set).toHaveBeenCalledWith('test-key', 'test-value', 'EX', 3600);
    });

    it('gets a key value', async () => {
        mockRedis.get.mockResolvedValue('test-value');
        const value = await service.get('test-key');
        expect(value).toBe('test-value');
        expect(mockRedis.get).toHaveBeenCalledWith('test-key');
    });

    it('deletes a key', async () => {
        await service.del('test-key');
        expect(mockRedis.del).toHaveBeenCalledWith('test-key');
    });

    it('revokes a session by session id', async () => {
        await service.revokeSession('session-id', 1800);

        expect(mockRedis.set).toHaveBeenCalledWith(
            'revoked:session:session-id',
            'true',
            'EX',
            1800,
        );
    });

    it('checks session revocation by session id with one Redis EXISTS lookup', async () => {
        mockRedis.exists.mockResolvedValue(1);

        await expect(service.isSessionRevoked('session-id')).resolves.toBe(true);
        expect(mockRedis.exists).toHaveBeenCalledTimes(1);
        expect(mockRedis.exists).toHaveBeenCalledWith('revoked:session:session-id');
        expect(mockRedis.get).not.toHaveBeenCalled();
    });

    it('returns false when session revocation key does not exist', async () => {
        mockRedis.exists.mockResolvedValue(0);

        await expect(service.isSessionRevoked('session-id')).resolves.toBe(false);
        expect(mockRedis.exists).toHaveBeenCalledWith('revoked:session:session-id');
    });

    it('should call recordSuccess on successful operation', async () => {
        mockRedis.get.mockResolvedValue('value');
        await service.get('key');
        expect(mockRedisService.recordSuccess).toHaveBeenCalled();
    });

    it('should call recordFailure and throw error when operation fails', async () => {
        mockRedis.get.mockRejectedValue(new Error('Redis Error'));
        await expect(service.get('key')).rejects.toThrow('Redis Error');
        expect(mockRedisService.recordFailure).toHaveBeenCalled();
    });
});
