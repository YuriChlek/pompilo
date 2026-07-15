import { Global, Module } from '@nestjs/common';
import { DRIZZLE_PROVIDER, DrizzleProvider } from './providers/drizzle.provider';
import { TransactionRepository } from '@/module-drizzle/repository/transaction.repository';

@Global()
@Module({
    providers: [...DrizzleProvider, TransactionRepository],
    exports: [DRIZZLE_PROVIDER, TransactionRepository],
})
export class DrizzleModule {}
