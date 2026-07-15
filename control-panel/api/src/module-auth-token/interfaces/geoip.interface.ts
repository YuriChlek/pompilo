export interface GeoIpMetadata {
    country: string;
    region: string;
    city: string;
}

export interface GeoIpProvider {
    lookup(ipAddress: string): Promise<GeoIpMetadata>;
}
