import * as argon2 from 'argon2';

export class Argon2HashUtil {
    static async hash(data: string): Promise<string> {
        return argon2.hash(data, {
            type: argon2.argon2id,
            memoryCost: 2 ** 16, // 64 MB
            timeCost: 3,
            parallelism: 1,
        });
    }

    static async compare(data: string, hash: string): Promise<boolean> {
        return argon2.verify(hash, data);
    }
}
