import { SetMetadata } from '@nestjs/common';
import { EMAIL_FLOW_RATE_LIMIT_METADATA_KEY } from '@/common/rate-limiting/constants/email-flow-rate-limit.constants';
import type { EmailFlowRateLimitOptions } from '@/common/rate-limiting/interfaces/email-flow-rate-limit.interfaces';

export const EmailFlowRateLimit = (options: EmailFlowRateLimitOptions) =>
    SetMetadata(EMAIL_FLOW_RATE_LIMIT_METADATA_KEY, options);
