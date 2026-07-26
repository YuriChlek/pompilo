import { runCli } from '../../../../scripts/cli';
import { seedDemoData } from '../../../../scripts/cli/operations/demo-data/seed-demo-data';

jest.mock('../../../../scripts/cli/operations/demo-data/seed-demo-data', () => ({
    seedDemoData: jest.fn(),
}));

describe('Demo Data CLI Commands', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('demo-data:push should call seedDemoData', async () => {
        const exitCode = await runCli(['demo-data:push']);
        expect(exitCode).toBe(0);
        expect(seedDemoData).toHaveBeenCalled();
    });

    it('demo-data:push should return 1 and log error when seedDemoData fails', async () => {
        (seedDemoData as jest.Mock).mockRejectedValueOnce(new Error('Seeding failed'));
        const exitCode = await runCli(['demo-data:push']);
        expect(exitCode).toBe(1);
    });
});
