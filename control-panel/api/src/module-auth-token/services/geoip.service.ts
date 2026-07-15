import { Inject, Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { GeoIpMetadata, GeoIpProvider } from '@/module-auth-token/interfaces/geoip.interface';

@Injectable()
export class GeoIpService {
    private readonly logger = new Logger(GeoIpService.name);
    private readonly timeoutMs: number;

    constructor(
        @Inject('GEOIP_PROVIDER')
        private readonly geoIpProvider: GeoIpProvider,
        private readonly configService: ConfigService,
    ) {
        this.timeoutMs = this.configService.get<number>('GEOIP_TIMEOUT_MS', 1000);
    }

    async lookup(ipAddress: string): Promise<GeoIpMetadata> {
        try {
            const metadata = await this.withTimeout(
                this.geoIpProvider.lookup(ipAddress),
                this.timeoutMs,
                `GeoIP lookup timed out after ${this.timeoutMs}ms`,
            );

            return {
                ...metadata,
                country: this.normalizeCountry(metadata.country),
            };
        } catch (error) {
            this.logger.warn(
                `GeoIP lookup failed/timed out for IP ${ipAddress}. Falling back to unknown location. Error: ${
                    error instanceof Error ? error.message : String(error)
                }`,
            );
            return {
                country: 'ZZ',
                region: 'unknown',
                city: 'unknown',
            };
        }
    }

    private normalizeCountry(country: string): string {
        return /^[a-z]{2}$/i.test(country) ? country.toUpperCase() : 'ZZ';
    }

    private withTimeout<T>(
        promise: Promise<T>,
        timeoutMs: number,
        errorMessage: string,
    ): Promise<T> {
        let timeoutId: NodeJS.Timeout;
        const timeoutPromise = new Promise<never>((_, reject) => {
            timeoutId = setTimeout(() => {
                reject(new Error(errorMessage));
            }, timeoutMs);
        });

        return Promise.race([
            promise.then(result => {
                clearTimeout(timeoutId);
                return result;
            }),
            timeoutPromise,
        ]);
    }
}
