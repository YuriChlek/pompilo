import { LOGGER_CHANNELS } from '@/module-logger/constants/logger.constants';

describe('LOGGER_CHANNELS', () => {
    it('defines the standard logger channels for module-level injection', () => {
        expect(LOGGER_CHANNELS).toEqual([
            'system',
            'auth',
            'auth-token',
            'mail',
            'data-patch',
        ]);
    });
});
