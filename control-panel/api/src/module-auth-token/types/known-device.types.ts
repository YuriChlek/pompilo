export type KnownDeviceMetadataUpdate = {
    trustedAt?: Date | null;
    trustExpiresAt?: Date | null;
    lastSeenAt?: Date;
    lastIpAddress?: string | null;
    lastCountry?: string | null;
    lastRegion?: string | null;
    lastCity?: string | null;
    lastUserAgent?: string | null;
};

export type KnownDeviceMetadata = {
    lastIpAddress?: string | null;
    lastCountry?: string | null;
    lastRegion?: string | null;
    lastCity?: string | null;
    lastUserAgent?: string | null;
};
