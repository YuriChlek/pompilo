import { SetMetadata } from '@nestjs/common';
import {
    ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY,
    AdminMailActionRateLimitOptions,
} from '@/module-mail/constants/admin-mail-action-rate-limit.constants';

export const AdminMailActionRateLimit = (options: AdminMailActionRateLimitOptions) =>
    SetMetadata(ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY, options);
