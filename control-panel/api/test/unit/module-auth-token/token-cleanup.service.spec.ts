import { Test, TestingModule } from '@nestjs/testing';
import { ScheduleModule, SchedulerRegistry } from '@nestjs/schedule';
import { TokenCleanupService } from '@/module-auth-token/services/token-cleanup.service';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';

describe('TokenCleanupService', () => {
    let service: TokenCleanupService;
    let mockAuthTokenRepository: jest.Mocked<AuthTokenRepository>;
    let mockReauthConfirmationService: jest.Mocked<ReauthConfirmationService>;
    let mockSessionRepository: jest.Mocked<SessionRepository>;
    let mockKnownDeviceRepository: jest.Mocked<KnownDeviceRepository>;
    let mockLoginChallengeRepository: jest.Mocked<LoginChallengeRepository>;
    let mockSecurityEventService: jest.Mocked<SecurityEventService>;

    beforeEach(() => {
        mockAuthTokenRepository = {
            deleteExpiredTokens: jest.fn(),
        } as unknown as jest.Mocked<AuthTokenRepository>;

        mockReauthConfirmationService = {
            expireReauthConfirmations: jest.fn(),
        } as unknown as jest.Mocked<ReauthConfirmationService>;

        mockSessionRepository = {
            deleteExpiredSessions: jest.fn(),
        } as unknown as jest.Mocked<SessionRepository>;

        mockKnownDeviceRepository = {
            deleteExpiredKnownDevices: jest.fn(),
        } as unknown as jest.Mocked<KnownDeviceRepository>;

        mockLoginChallengeRepository = {
            deleteExpiredChallenges: jest.fn(),
        } as unknown as jest.Mocked<LoginChallengeRepository>;

        mockSecurityEventService = {
            deleteExpiredEvents: jest.fn(),
        } as unknown as jest.Mocked<SecurityEventService>;

        service = new TokenCleanupService(
            mockAuthTokenRepository,
            mockReauthConfirmationService,
            mockSessionRepository,
            mockKnownDeviceRepository,
            mockLoginChallengeRepository,
            mockSecurityEventService,
        );

        jest.clearAllMocks();
    });

    it('should successfully run cron job and log results', async () => {
        mockAuthTokenRepository.deleteExpiredTokens.mockResolvedValue(5);
        mockReauthConfirmationService.expireReauthConfirmations.mockResolvedValue(10);
        mockSessionRepository.deleteExpiredSessions.mockResolvedValue(15);
        mockKnownDeviceRepository.deleteExpiredKnownDevices.mockResolvedValue(20);
        mockLoginChallengeRepository.deleteExpiredChallenges.mockResolvedValue(25);
        mockSecurityEventService.deleteExpiredEvents.mockResolvedValue(30);

        await service.handleCron();

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockAuthTokenRepository.deleteExpiredTokens).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockReauthConfirmationService.expireReauthConfirmations).toHaveBeenCalledWith(
            24 * 60 * 60 * 1000,
        );
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockLoginChallengeRepository.deleteExpiredChallenges).toHaveBeenCalledWith(
            24 * 60 * 60 * 1000,
        );
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSessionRepository.deleteExpiredSessions).toHaveBeenCalledWith(
            30 * 24 * 60 * 60 * 1000,
        );
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockKnownDeviceRepository.deleteExpiredKnownDevices).toHaveBeenCalledWith(
            30 * 24 * 60 * 60 * 1000,
        );
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSecurityEventService.deleteExpiredEvents).toHaveBeenCalledWith(
            90 * 24 * 60 * 60 * 1000,
        );
    });

    it('should catch errors and log them without crashing', async () => {
        mockAuthTokenRepository.deleteExpiredTokens.mockRejectedValue(
            new Error('AuthToken DB error'),
        );
        mockReauthConfirmationService.expireReauthConfirmations.mockRejectedValue(
            new Error('Reauth DB error'),
        );
        mockSessionRepository.deleteExpiredSessions.mockRejectedValue(
            new Error('Session DB error'),
        );
        mockKnownDeviceRepository.deleteExpiredKnownDevices.mockRejectedValue(
            new Error('KnownDevice DB error'),
        );
        mockLoginChallengeRepository.deleteExpiredChallenges.mockRejectedValue(
            new Error('LoginChallenge DB error'),
        );
        mockSecurityEventService.deleteExpiredEvents.mockRejectedValue(
            new Error('SecurityEvent DB error'),
        );

        await expect(service.handleCron()).resolves.toBeUndefined();

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockAuthTokenRepository.deleteExpiredTokens).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockReauthConfirmationService.expireReauthConfirmations).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockLoginChallengeRepository.deleteExpiredChallenges).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSessionRepository.deleteExpiredSessions).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockKnownDeviceRepository.deleteExpiredKnownDevices).toHaveBeenCalled();
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSecurityEventService.deleteExpiredEvents).toHaveBeenCalled();
    });

    it('should be idempotent when the scheduled cleanup job runs repeatedly', async () => {
        mockAuthTokenRepository.deleteExpiredTokens
            .mockResolvedValueOnce(5)
            .mockResolvedValueOnce(0);
        mockReauthConfirmationService.expireReauthConfirmations
            .mockResolvedValueOnce(10)
            .mockResolvedValueOnce(0);
        mockSessionRepository.deleteExpiredSessions
            .mockResolvedValueOnce(15)
            .mockResolvedValueOnce(0);
        mockKnownDeviceRepository.deleteExpiredKnownDevices
            .mockResolvedValueOnce(20)
            .mockResolvedValueOnce(0);
        mockLoginChallengeRepository.deleteExpiredChallenges
            .mockResolvedValueOnce(25)
            .mockResolvedValueOnce(0);
        mockSecurityEventService.deleteExpiredEvents
            .mockResolvedValueOnce(30)
            .mockResolvedValueOnce(0);

        await expect(service.handleCron()).resolves.toBeUndefined();
        await expect(service.handleCron()).resolves.toBeUndefined();

        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockAuthTokenRepository.deleteExpiredTokens).toHaveBeenCalledTimes(2);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockReauthConfirmationService.expireReauthConfirmations).toHaveBeenCalledTimes(2);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockLoginChallengeRepository.deleteExpiredChallenges).toHaveBeenCalledTimes(2);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSessionRepository.deleteExpiredSessions).toHaveBeenCalledTimes(2);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockKnownDeviceRepository.deleteExpiredKnownDevices).toHaveBeenCalledTimes(2);
        // eslint-disable-next-line @typescript-eslint/unbound-method
        expect(mockSecurityEventService.deleteExpiredEvents).toHaveBeenCalledTimes(2);
    });

    it('should compile within a NestJS module and register the cron job', async () => {
        const module: TestingModule = await Test.createTestingModule({
            imports: [ScheduleModule.forRoot()],
            providers: [
                TokenCleanupService,
                { provide: AuthTokenRepository, useValue: mockAuthTokenRepository },
                { provide: ReauthConfirmationService, useValue: mockReauthConfirmationService },
                { provide: SessionRepository, useValue: mockSessionRepository },
                { provide: KnownDeviceRepository, useValue: mockKnownDeviceRepository },
                { provide: LoginChallengeRepository, useValue: mockLoginChallengeRepository },
                { provide: SecurityEventService, useValue: mockSecurityEventService },
            ],
        }).compile();

        await module.init();

        const schedulerRegistry = module.get<SchedulerRegistry>(SchedulerRegistry);
        const cronJobs = schedulerRegistry.getCronJobs();
        expect(cronJobs.size).toBe(1);

        const job = schedulerRegistry.getCronJob('token-cleanup-job');
        expect(job).toBeDefined();
        expect(typeof job.start).toBe('function');

        await module.close();
    });
});
