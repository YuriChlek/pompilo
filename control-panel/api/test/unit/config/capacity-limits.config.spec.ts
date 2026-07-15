import { ConfigService } from '@nestjs/config';
import { getDbPoolMax, INITIAL_CAPACITY_LIMITS } from '@/config/capacity-limits.config';

describe('capacityLimitsConfig', () => {
    it('uses the initial per-replica DB pool limit by default', () => {
        const configService = new ConfigService({});

        expect(getDbPoolMax(configService)).toBe(INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica);
    });

    it('allows a smaller positive DB pool limit', () => {
        const configService = new ConfigService({ DB_POOL_MAX: '12' });

        expect(getDbPoolMax(configService)).toBe(12);
    });

    it('caps the DB pool at 30 connections per API replica', () => {
        const configService = new ConfigService({ DB_POOL_MAX: '60' });

        expect(getDbPoolMax(configService)).toBe(30);
    });

    it.each(['0', '-1', '2.5', 'invalid'])(
        'falls back to the initial DB pool limit for invalid value %s',
        value => {
            const configService = new ConfigService({ DB_POOL_MAX: value });

            expect(getDbPoolMax(configService)).toBe(
                INITIAL_CAPACITY_LIMITS.dbPoolMaxPerApiReplica,
            );
        },
    );

    it('keeps the initial mail throughput targets internally consistent', () => {
        const windowsPerMinute = 60_000 / INITIAL_CAPACITY_LIMITS.mailQueueLimiterDurationMs;

        expect(INITIAL_CAPACITY_LIMITS.criticalEmailRequestsPerSecond * windowsPerMinute).toBe(
            INITIAL_CAPACITY_LIMITS.emailDeliveriesPerMinute,
        );
    });
});
