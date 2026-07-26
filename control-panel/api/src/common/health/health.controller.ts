import { Controller, Get, ServiceUnavailableException } from '@nestjs/common';
import { RedisService } from '@/common/redis/redis.service';
import { ApiOperation, ApiTags } from '@nestjs/swagger';

@ApiTags('Health')
@Controller('health')
export class HealthController {
    constructor(private readonly redisService: RedisService) {}

    @Get('readiness')
    @ApiOperation({ summary: 'Application readiness check' })
    async checkReadiness() {
        if (!this.redisService.isHealthy()) {
            throw new ServiceUnavailableException('Redis consecutive failures limit reached');
        }

        try {
            const client = this.redisService.getClient();
            const result = await client.ping();
            if (result !== 'PONG') {
                throw new Error('Ping failed');
            }
            this.redisService.recordSuccess();
        } catch {
            this.redisService.recordFailure();
            throw new ServiceUnavailableException('Redis is unreachable');
        }

        return { status: 'healthy' };
    }
}
