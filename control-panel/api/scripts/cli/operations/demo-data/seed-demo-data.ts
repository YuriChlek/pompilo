/**
 * Seeds demo identity users.
 *
 * Run from api/: npm run cli -- demo-data:push
 */
import { seedDemoUsers } from './seed-users';

export async function seedDemoData(): Promise<void> {
    const userSummary = await seedDemoUsers();
    if (userSummary.failed > 0) {
        throw new Error(`Seeding demo users failed with ${userSummary.failed} errors`);
    }
}
