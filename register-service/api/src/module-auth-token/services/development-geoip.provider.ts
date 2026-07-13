import { Injectable } from '@nestjs/common';
import { GeoIpMetadata, GeoIpProvider } from '@/module-auth-token/interfaces/geoip.interface';

@Injectable()
export class DevelopmentGeoIpProvider implements GeoIpProvider {
    async lookup(ipAddress: string): Promise<GeoIpMetadata> {
        await Promise.resolve(ipAddress);
        return {
            country: 'ZZ',
            region: 'unknown',
            city: 'unknown',
        };
    }
}
