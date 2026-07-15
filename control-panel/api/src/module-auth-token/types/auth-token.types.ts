import type { TokenUserPayload } from '@/module-auth-token/types/token-db.types';

export type RefreshTokenVerificationResult =
    | {
          verified: true;
          tokenId: string;
          sessionId: string;
          user: TokenUserPayload;
          isGrace: boolean;
      }
    | { verified: false; user?: never };
