import { Injectable } from '@nestjs/common';
import { Argon2HashUtil } from '@/common/utils/hash.util';

@Injectable()
export class UserPasswordService {
    async hashPassword(password: string): Promise<string> {
        return await Argon2HashUtil.hash(password);
    }

    async comparePassword(password: string, hash: string): Promise<boolean> {
        return await Argon2HashUtil.compare(password, hash);
    }
}
