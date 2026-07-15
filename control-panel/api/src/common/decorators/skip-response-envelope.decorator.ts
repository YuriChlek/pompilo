import { SetMetadata } from '@nestjs/common';

export const SKIP_RESPONSE_ENVELOPE_KEY = 'skipResponseEnvelope';

export const SkipResponseEnvelope = () => SetMetadata(SKIP_RESPONSE_ENVELOPE_KEY, true);
