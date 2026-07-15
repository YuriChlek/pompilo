import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import { ConfigService } from '@nestjs/config';
import { GeoIpProvider, GeoIpMetadata } from '@/module-auth-token/interfaces/geoip.interface';

describe('GeoIpService', () => {
    let service: GeoIpService;
    let mockProvider: jest.Mocked<GeoIpProvider>;
    let configService: {
        get: jest.Mock;
    };

    beforeEach(() => {
        mockProvider = {
            lookup: jest.fn(),
        };

        configService = {
            get: jest.fn().mockImplementation((key: string, defaultValue: unknown) => {
                if (key === 'GEOIP_TIMEOUT_MS') {
                    return 50; // Use small timeout for testing
                }
                return defaultValue;
            }),
        };

        service = new GeoIpService(mockProvider, configService as unknown as ConfigService);
    });

    it('returns resolved location metadata when provider succeeds', async () => {
        const expectedMetadata: GeoIpMetadata = {
            country: 'UA',
            region: 'Kyiv Oblast',
            city: 'Kyiv',
        };
        mockProvider.lookup.mockResolvedValue(expectedMetadata);

        const result = await service.lookup('82.207.35.41');

        expect(result).toEqual(expectedMetadata);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockProvider.lookup).toHaveBeenCalledWith('82.207.35.41');
    });

    it('normalizes non-ISO country names to the unknown country code', async () => {
        mockProvider.lookup.mockResolvedValue({
            country: 'Ukraine',
            region: 'Kyiv Oblast',
            city: 'Kyiv',
        });

        const result = await service.lookup('82.207.35.41');

        expect(result).toEqual({
            country: 'ZZ',
            region: 'Kyiv Oblast',
            city: 'Kyiv',
        });
    });

    it('returns unknown fallback and logs warning when provider throws an error', async () => {
        mockProvider.lookup.mockRejectedValue(new Error('Connection error'));

        const result = await service.lookup('82.207.35.41');

        expect(result).toEqual({
            country: 'ZZ',
            region: 'unknown',
            city: 'unknown',
        });
    });

    it('returns unknown fallback and logs warning when provider times out', async () => {
        // Mock provider lookup that takes longer than the timeout limit of 50ms
        mockProvider.lookup.mockImplementation(
            () =>
                new Promise<GeoIpMetadata>(resolve => {
                    setTimeout(() => {
                        resolve({
                            country: 'UA',
                            region: 'Kyiv Oblast',
                            city: 'Kyiv',
                        });
                    }, 200);
                }),
        );

        const result = await service.lookup('82.207.35.41');

        expect(result).toEqual({
            country: 'ZZ',
            region: 'unknown',
            city: 'unknown',
        });
    });
});
