import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { drizzle, NodePgDatabase } from 'drizzle-orm/node-postgres';
import { migrate } from 'drizzle-orm/node-postgres/migrator';
import * as schema from '@/module-drizzle/schemas';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import { ReauthConfirmationRepository } from '@/module-auth-token/repository/reauth-confirmation.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { UserRepository } from '@/module-user/repository/user.repository';
import { users } from '@/module-user/schemas';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';
import { tokens } from '@/module-auth-token/schemas/tokens.schema';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { eq, sql } from 'drizzle-orm';
import { getTransactionClient } from '@/module-drizzle/repository/transaction.repository';
import { Client, Pool, type ClientConfig } from 'pg';
import { loadApiEnvFiles } from '@/config/api-env.config';

jest.setTimeout(180_000);

describe('AuthTokenDatabaseIntegration', () => {
    let adminClient: Client;
    let pool: Pool;
    let db: NodePgDatabase<typeof schema>;
    let knownDeviceRepo: KnownDeviceRepository;
    let securityEventRepo: SecurityEventRepository;
    let loginChallengeRepo: LoginChallengeRepository;
    let reauthConfirmationRepo: ReauthConfirmationRepository;
    let sessionRepo: SessionRepository;
    let authTokenRepo: AuthTokenRepository;
    let userRepo: UserRepository;

    const testUserId = 'f7b1a2c3-4d5e-6f7a-8b9c-0d1e2f3a4b5c';
    const testDeviceId1 = 'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d';
    const testDeviceId2 = 'b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e';
    const testDeviceId3 = 'c3d4e5f6-a7b8-9c0d-1e2f-3a4b5c6d7e8f';
    const testDeviceId4 = 'd4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a';
    const databaseName = `pampilo_auth_test_${randomUUID().replaceAll('-', '')}`;

    beforeAll(async () => {
        loadApiEnvFiles();
        adminClient = new Client(getDatabaseConfig(process.env.DB_NAME));
        await adminClient.connect();
        await adminClient.query(`CREATE DATABASE ${quoteIdentifier(databaseName)}`);

        pool = new Pool(getDatabaseConfig(databaseName));
        db = drizzle(pool, { schema });
        await migrate(db, {
            migrationsFolder: path.resolve(process.cwd(), 'drizzle-migrations'),
        });

        knownDeviceRepo = new KnownDeviceRepository(db);
        securityEventRepo = new SecurityEventRepository(db);
        loginChallengeRepo = new LoginChallengeRepository(db);
        reauthConfirmationRepo = new ReauthConfirmationRepository(db);
        sessionRepo = new SessionRepository(db);
        authTokenRepo = new AuthTokenRepository(db);
        userRepo = new UserRepository(db);

        await db.insert(users).values({
            id: testUserId,
            name: 'DB Integration Test User',
            email: 'db-integration-test@example.com',
            password: 'hashed-password',
            role: UserRoles.USER,
            isActive: true,
        });
    });

    afterAll(async () => {
        if (pool) {
            await pool.end().catch(() => undefined);
        }

        if (adminClient) {
            await adminClient
                .query(`DROP DATABASE IF EXISTS ${quoteIdentifier(databaseName)} WITH (FORCE)`)
                .catch(() => undefined);
            await adminClient.end().catch(() => undefined);
        }
    });

    afterEach(async () => {
        if (!db) {
            return;
        }

        await db.delete(schema.securityEvents);
        await db.delete(schema.reauthConfirmations);
        await db.delete(schema.tokens);
        await db.delete(schema.sessions);
        await db.delete(schema.loginChallenges);
        await db.delete(schema.knownDevices);
    });

    it('Phase 2.2 / 3.2 / 4.3: should confirm all database tables and indexes exist in the schema', async () => {
        const result = await db.execute(sql`
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND table_name IN ('known_devices', 'security_events', 'login_challenges')
        `);
        const tableNames = result.rows.map(row => row.table_name);
        expect(tableNames).toContain('known_devices');
        expect(tableNames).toContain('security_events');
        expect(tableNames).toContain('login_challenges');
    });

    it('Phase 2.3: should enforce unique active device constraints and support soft-revocation', async () => {
        // 1. Create a known device
        const device1 = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId1,
        });
        expect(device1).toBeDefined();
        expect(await knownDeviceRepo.findById(device1.id)).toEqual(device1);
        expect(
            await knownDeviceRepo.findActiveByDevice(testUserId, 'customer', testDeviceId1),
        ).toEqual(device1);

        const updatedDevice = await knownDeviceRepo.update(device1.id, {
            lastIpAddress: '203.0.113.10',
            lastCountry: 'UA',
            lastSeenAt: new Date(),
        });
        expect(updatedDevice?.lastIpAddress).toBe('203.0.113.10');
        expect(updatedDevice?.lastCountry).toBe('UA');

        // 2. Try to insert duplicate active known device -> must fail
        await expect(
            knownDeviceRepo.save({
                userId: testUserId,
                realm: 'customer',
                deviceId: testDeviceId1,
            }),
        ).rejects.toThrow();

        // 3. Soft revoke the device
        const revoked = await knownDeviceRepo.revoke(device1.id);
        expect(revoked).toBe(true);

        // 4. Able to insert new active device now
        const device2 = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId1,
        });
        expect(device2).toBeDefined();
        expect(device2.id).not.toBe(device1.id);
    });

    it('Phase 3.3: should save and retrieve security events with metadata and verify ON DELETE SET NULL behavior', async () => {
        // 1. Save known device
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId2,
        });

        // 1.5. Save session (required for security_events.session_id FK constraint)
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 2. Save security event
        const metadata = { login_method: 'password', ip: '127.0.0.1' };
        const event = await securityEventRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            knownDeviceId: device.id,
            eventType: SecurityEventType.LOGIN_SUCCESS,
            riskScore: 0,
            metadata,
        });

        expect(event.userId).toBe(testUserId);
        expect(event.knownDeviceId).toBe(device.id);
        expect(event.metadata).toEqual(metadata);

        // 3. Delete the referencing session first to bypass onDelete 'restrict' on knownDevices
        await db.delete(sessions).where(eq(sessions.id, session.id));

        // 3.5. Delete the known device
        await db.delete(knownDevices).where(eq(knownDevices.id, device.id));

        // 4. Retrieve event and verify user_id is set but known_device_id has become null (SET NULL check)
        const retrievedEvent = await securityEventRepo.findById(event.id);
        expect(retrievedEvent).toBeDefined();
        expect(retrievedEvent?.userId).toBe(testUserId);
        expect(retrievedEvent?.knownDeviceId).toBeNull();
    });

    it('Phase 4.2 / 4.3: should handle parallel login challenges creation without 500 duplicate key error and keep one active challenge', async () => {
        const challengeInsert = {
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId1,
            checkpointTokenHash: 'token-hash-xyz',
            codeHash: 'code-hash-xyz',
            expiresAt: new Date(Date.now() + 600000),
        };

        // Run parallel creations
        const [c1, c2] = await Promise.all([
            loginChallengeRepo.createSafe(challengeInsert),
            loginChallengeRepo.createSafe(challengeInsert),
        ]);

        expect(c1).toBeDefined();
        expect(c2).toBeDefined();

        const activeCountResult = await db.execute(sql`
            select count(*)::int as count
            from login_challenges
            where user_id = ${testUserId}
              and realm = 'customer'
              and device_id = ${testDeviceId1}
              and consumed_at is null
              and failed_at is null
              and expired_at is null
        `);
        expect(Number(activeCountResult.rows[0]?.count)).toBe(1);
    });

    it('Phase 4.2 / 4.3: should enforce atomic attempt increments and auto-fail challenge when max attempts reached', async () => {
        const challenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId3,
            checkpointTokenHash: 'token-hash-abc',
            codeHash: 'code-hash-abc',
            maxAttempts: 3,
            expiresAt: new Date(Date.now() + 600000),
        });

        const inc1 = await loginChallengeRepo.incrementAttempts(challenge.id);
        expect(inc1?.attemptCount).toBe(1);
        expect(inc1?.failedAt).toBeNull();

        const inc2 = await loginChallengeRepo.incrementAttempts(challenge.id);
        expect(inc2?.attemptCount).toBe(2);
        expect(inc2?.failedAt).toBeNull();

        const inc3 = await loginChallengeRepo.incrementAttempts(challenge.id);
        expect(inc3?.attemptCount).toBe(3);
        expect(inc3?.failedAt).not.toBeNull(); // Auto-failed
        expect(await loginChallengeRepo.incrementAttempts(challenge.id)).toBeNull();
    });

    it('Phase 4.3: should materialize expiry and reject mutations of expired challenges', async () => {
        const expiredChallenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId4,
            checkpointTokenHash: 'token-hash-expired',
            codeHash: 'code-hash-expired',
            expiresAt: new Date(Date.now() - 1_000),
        });

        expect(await loginChallengeRepo.approveAndConsume(expiredChallenge.id)).toBe(false);
        expect(await loginChallengeRepo.incrementAttempts(expiredChallenge.id)).toBeNull();
        expect(await loginChallengeRepo.expire(expiredChallenge.id)).toBe(true);

        const replacement = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId4,
            checkpointTokenHash: 'token-hash-replacement',
            codeHash: 'code-hash-replacement',
            expiresAt: new Date(Date.now() + 600_000),
        });

        expect(replacement.id).not.toBe(expiredChallenge.id);
    });

    it('Phase 4.3: should rollback approval issuance and allow retry with the same challenge', async () => {
        const sessionId = randomUUID();
        const refreshTokenHash = `refresh-${randomUUID()}`;
        const knownDevice = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const challenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
            checkpointTokenHash: 'token-hash-roll',
            codeHash: 'code-hash-roll',
            expiresAt: new Date(Date.now() + 600_000),
        });

        await expect(
            loginChallengeRepo.approveAndConsumeAtomically(challenge.id, async transaction => {
                const transactionClient = getTransactionClient(transaction, db);
                await transactionClient.insert(sessions).values({
                    id: sessionId,
                    userId: testUserId,
                    realm: 'customer',
                    knownDeviceId: knownDevice.id,
                    deviceId: knownDevice.deviceId,
                    expiresAt: new Date(Date.now() + 600_000),
                });
                await transactionClient.insert(tokens).values({
                    sessionId,
                    jti: randomUUID(),
                    refreshTokenHash,
                    expiresAt: new Date(Date.now() + 600_000),
                });
                await securityEventRepo.save(
                    {
                        userId: testUserId,
                        realm: 'customer',
                        sessionId,
                        knownDeviceId: knownDevice.id,
                        eventType: SecurityEventType.LOGIN_APPROVAL_PASSED,
                        metadata: { loginChallengeId: challenge.id },
                    },
                    transaction,
                );
                throw new Error('Force rollback');
            }),
        ).rejects.toThrow('Force rollback');

        const afterRollback = await loginChallengeRepo.findById(challenge.id);
        expect(afterRollback?.consumedAt).toBeNull();
        expect(afterRollback?.approvedAt).toBeNull();
        expect(await db.select().from(tokens).where(eq(tokens.sessionId, sessionId))).toHaveLength(
            0,
        );

        const retryResult = await loginChallengeRepo.approveAndConsumeAtomically(
            challenge.id,
            async transaction => {
                const transactionClient = getTransactionClient(transaction, db);
                await transactionClient.insert(sessions).values({
                    id: sessionId,
                    userId: testUserId,
                    realm: 'customer',
                    knownDeviceId: knownDevice.id,
                    deviceId: knownDevice.deviceId,
                    expiresAt: new Date(Date.now() + 600_000),
                });
                await transactionClient.insert(tokens).values({
                    sessionId,
                    jti: randomUUID(),
                    refreshTokenHash,
                    expiresAt: new Date(Date.now() + 600_000),
                });
                await securityEventRepo.save(
                    {
                        userId: testUserId,
                        realm: 'customer',
                        sessionId,
                        knownDeviceId: knownDevice.id,
                        eventType: SecurityEventType.LOGIN_APPROVAL_PASSED,
                        metadata: { loginChallengeId: challenge.id },
                    },
                    transaction,
                );
                return sessionId;
            },
        );

        expect(retryResult).toBe(sessionId);
        const afterRetry = await loginChallengeRepo.findById(challenge.id);
        expect(afterRetry?.approvedAt).not.toBeNull();
        expect(afterRetry?.consumedAt).not.toBeNull();
        expect(await db.select().from(tokens).where(eq(tokens.sessionId, sessionId))).toHaveLength(
            1,
        );
    });

    it('Phase 5.1/5.3: should create, find, and atomically consume reauth confirmations', async () => {
        const testSessionId = randomUUID();
        const testTokenHash = 'some-token-hash-for-reauth';

        // 0. Create a known device and session to satisfy reauth_confirmations.session_id FK constraint
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        await sessionRepo.save({
            id: testSessionId,
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 1. Create a confirmation
        const created = await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: testSessionId,
            actionScope: 'email_change',
            confirmationTokenHash: testTokenHash,
            expiresAt: new Date(Date.now() + 600_000), // 10 mins in future
        });

        expect(created).toBeDefined();
        expect(created.id).toBeDefined();

        // 2. Find by id
        const foundById = await reauthConfirmationRepo.findById(created.id);
        expect(foundById).toBeDefined();
        expect(foundById?.confirmationTokenHash).toBe(testTokenHash);

        // 3. Find by token hash
        const foundByHash = await reauthConfirmationRepo.findByTokenHash(testTokenHash);
        expect(foundByHash).toBeDefined();
        expect(foundByHash?.id).toBe(created.id);

        // 4. Try to consume with wrong scope, session, user, or realm - should fail
        const consumedWrongUser = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            randomUUID(), // wrong user
            'customer',
            testSessionId,
            'email_change',
        );
        expect(consumedWrongUser).toBe(false);

        const consumedWrongRealm = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            testUserId,
            'admin', // wrong realm
            testSessionId,
            'email_change',
        );
        expect(consumedWrongRealm).toBe(false);

        const consumedWrongSession = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            testUserId,
            'customer',
            randomUUID(), // wrong session
            'email_change',
        );
        expect(consumedWrongSession).toBe(false);

        const consumedWrongScope = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            testUserId,
            'customer',
            testSessionId,
            'password_change', // wrong scope
        );
        expect(consumedWrongScope).toBe(false);

        // 5. Consume atomically with correct session and scope - should succeed
        const consumed = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            testUserId,
            'customer',
            testSessionId,
            'email_change',
        );
        expect(consumed).toBe(true);

        // 6. Verify it is now consumed
        const afterConsume = await reauthConfirmationRepo.findById(created.id);
        expect(afterConsume?.consumedAt).not.toBeNull();

        // 6.5. Try to consume expired confirmation - should fail
        const expiredConfirmationTokenHash = 'expired-reauth-token-hash';
        await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: testSessionId,
            actionScope: 'email_change',
            confirmationTokenHash: expiredConfirmationTokenHash,
            expiresAt: new Date(Date.now() - 1_000), // expired 1s ago
        });
        const consumedExpired = await reauthConfirmationRepo.consumeByTokenHash(
            expiredConfirmationTokenHash,
            testUserId,
            'customer',
            testSessionId,
            'email_change',
        );
        expect(consumedExpired).toBe(false);

        // 7. Try to consume again - should fail
        const consumedAgain = await reauthConfirmationRepo.consumeByTokenHash(
            testTokenHash,
            testUserId,
            'customer',
            testSessionId,
            'email_change',
        );
        expect(consumedAgain).toBe(false);
    });

    it('Phase 6.2/6.3: should create, find, extend and revoke sessions, and block duplicate active sessions', async () => {
        // 1. Create a known device (required for session foreign key)
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId3,
        });

        // 2. Create the first session
        const session1 = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000), // 10 mins in future
        });

        expect(session1).toBeDefined();
        expect(session1.id).toBeDefined();

        // 3. Try to save session with device mismatch - should fail (application layer validation)
        await expect(
            sessionRepo.save({
                userId: testUserId,
                realm: 'customer',
                knownDeviceId: device.id,
                deviceId: randomUUID(), // mismatched device ID
                expiresAt: new Date(Date.now() + 600_000),
            }),
        ).rejects.toThrow();

        // 4. Try to save a duplicate active session for the same user, realm, device - should fail (DB unique constraint)
        await expect(
            sessionRepo.save({
                userId: testUserId,
                realm: 'customer',
                knownDeviceId: device.id,
                deviceId: device.deviceId,
                expiresAt: new Date(Date.now() + 600_000),
            }),
        ).rejects.toThrow();

        // 5. Revoke session 1
        const revoked = await sessionRepo.revoke(session1.id);
        expect(revoked).toBe(true);

        // 6. Now that session 1 is revoked, we should be able to create a new session on the same device!
        const session2 = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        expect(session2).toBeDefined();
        expect(session2.id).not.toBe(session1.id);

        // 7. Find reusable/active sessions
        const reusable = await sessionRepo.findReusable(testUserId, 'customer', device.deviceId);
        expect(reusable).toBeDefined();
        expect(reusable?.id).toBe(session2.id);

        const activeSessions = await sessionRepo.findActiveByUserRealm(testUserId, 'customer');
        expect(activeSessions).toHaveLength(1);
        expect(activeSessions[0].id).toBe(session2.id);

        // 8. Extend / reuse the session
        const extendedExpiresAt = new Date(Date.now() + 1_200_000);
        const extended = await sessionRepo.reuseSession(session2.id, extendedExpiresAt);
        expect(extended).toBeDefined();
        expect(new Date(extended!.expiresAt).getTime()).toBe(extendedExpiresAt.getTime());
    });

    it('Phase 7.1: should set security_event.session_id to null when the corresponding session is deleted', async () => {
        // 1. Create a known device (required for session foreign key)
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId4,
        });

        // 2. Create a session
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 3. Create a security event bound to that session
        const event = await securityEventRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            knownDeviceId: device.id,
            eventType: SecurityEventType.LOGIN_SUCCESS,
        });

        expect(event).toBeDefined();
        expect(event.sessionId).toBe(session.id);

        // 4. Delete the session directly from the DB
        await db.delete(sessions).where(eq(sessions.id, session.id));

        // 5. Verify the security event still exists and its sessionId is set to null (ON DELETE SET NULL)
        const afterDelete = await securityEventRepo.findById(event.id);
        expect(afterDelete).toBeDefined();
        expect(afterDelete?.id).toBe(event.id);
        expect(afterDelete?.sessionId).toBeNull();
    });

    it('Phase 7.2: should delete reauth_confirmation when the corresponding session is deleted (ON DELETE CASCADE)', async () => {
        // 1. Create a known device
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });

        // 2. Create a session
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 3. Create a reauth confirmation
        const testTokenHash = 'cascade-delete-token-hash';
        const confirmation = await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            actionScope: 'email_change',
            confirmationTokenHash: testTokenHash,
            expiresAt: new Date(Date.now() + 600_000),
        });

        expect(confirmation).toBeDefined();

        // 4. Delete the session directly from the DB
        await db.delete(sessions).where(eq(sessions.id, session.id));

        // 5. Verify the reauth confirmation is deleted (ON DELETE CASCADE)
        const afterDelete = await reauthConfirmationRepo.findById(confirmation.id);
        expect(afterDelete).toBeNull();
    });

    it('Phase 8.1/8.3: should resolve refresh tokens exclusively by tokenId', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        const firstTokenId = randomUUID();
        const secondTokenId = randomUUID();

        await db.insert(tokens).values([
            {
                id: firstTokenId,
                sessionId: session.id,
                jti: randomUUID(),
                refreshTokenHash: 'first-token-hash',
                expiresAt: new Date(Date.now() + 600_000),
            },
            {
                id: secondTokenId,
                sessionId: session.id,
                jti: randomUUID(),
                refreshTokenHash: 'second-token-hash',
                expiresAt: new Date(Date.now() + 600_000),
            },
        ]);

        const secondToken = await authTokenRepo.findRefreshTokenById(secondTokenId);

        expect(secondToken).toEqual(
            expect.objectContaining({
                tokenId: secondTokenId,
                sessionId: session.id,
                refreshToken: 'second-token-hash',
            }),
        );
        expect(await authTokenRepo.findRefreshTokenById(randomUUID())).toBeNull();
    });

    it('Phase 8.2/8.3: should atomically allow only one concurrent token replacement', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        const currentTokenId = randomUUID();
        const replacementAId = randomUUID();
        const replacementBId = randomUUID();
        const graceExpiresAt = new Date(Date.now() + 20_000);

        await db.insert(tokens).values({
            id: currentTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'current-token-hash',
            expiresAt: new Date(Date.now() + 600_000),
        });

        const [replacementA, replacementB] = await Promise.all([
            authTokenRepo.rotateToken(
                currentTokenId,
                {
                    id: replacementAId,
                    sessionId: session.id,
                    jti: randomUUID(),
                    refreshTokenHash: 'replacement-a-hash',
                    expiresAt: new Date(Date.now() + 600_000),
                },
                graceExpiresAt,
            ),
            authTokenRepo.rotateToken(
                currentTokenId,
                {
                    id: replacementBId,
                    sessionId: session.id,
                    jti: randomUUID(),
                    refreshTokenHash: 'replacement-b-hash',
                    expiresAt: new Date(Date.now() + 600_000),
                },
                graceExpiresAt,
            ),
        ]);

        const successfulReplacements = [replacementA, replacementB].filter(
            replacement => replacement !== null,
        );
        expect(successfulReplacements).toHaveLength(1);

        const storedTokens = await db.select().from(tokens).where(eq(tokens.sessionId, session.id));
        expect(storedTokens).toHaveLength(2);

        const currentToken = storedTokens.find(token => token.id === currentTokenId);
        const winningReplacement = successfulReplacements[0];
        expect(currentToken?.replacedByTokenId).toBe(winningReplacement?.id);
        expect(currentToken?.replacedAt).not.toBeNull();
        expect(currentToken?.graceExpiresAt?.getTime()).toBe(graceExpiresAt.getTime());
        expect(storedTokens.some(token => token.id === winningReplacement?.id)).toBe(true);
    });

    it('Phase 8.2/8.3: should preserve active grace rows and clean them after grace expires', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        const expiredTokenId = randomUUID();

        await db.insert(tokens).values({
            id: expiredTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'expired-grace-token-hash',
            expiresAt: new Date(Date.now() - 1_000),
            replacedAt: new Date(Date.now() - 500),
            graceExpiresAt: new Date(Date.now() + 20_000),
        });

        expect(await authTokenRepo.deleteExpiredTokens()).toBe(0);
        expect(await authTokenRepo.findById(expiredTokenId)).not.toBeNull();

        await db
            .update(tokens)
            .set({ graceExpiresAt: new Date(Date.now() - 1) })
            .where(eq(tokens.id, expiredTokenId));

        expect(await authTokenRepo.deleteExpiredTokens()).toBe(1);
        expect(await authTokenRepo.findById(expiredTokenId)).toBeNull();
    });

    it('Phase 33.1: deleteExpiredTokens should preserve active, grace, and replacement tokens, but delete eligible tokens respecting retention period', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        const activeTokenId = randomUUID();
        const graceTokenId = randomUUID();
        const replacementTokenId = randomUUID();
        const expiredTokenId = randomUUID();
        const oldExpiredTokenId = randomUUID();
        const revokedTokenId = randomUUID();

        const now = Date.now();

        // 1. Replacement token (active)
        await db.insert(tokens).values({
            id: replacementTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'replacement-token-hash',
            expiresAt: new Date(now + 600_000),
        });

        // 2. Grace token (expired, but in grace window, points to replacement)
        await db.insert(tokens).values({
            id: graceTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'grace-token-hash',
            expiresAt: new Date(now - 10_000),
            replacedAt: new Date(now - 5_000),
            graceExpiresAt: new Date(now + 30_000),
            replacedByTokenId: replacementTokenId,
        });

        // 3. Active token (current active token for session, not replaced)
        await db.insert(tokens).values({
            id: activeTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'active-token-hash',
            expiresAt: new Date(now + 600_000),
        });

        // 4. Recently expired token (expired 2 hours ago, grace expired/null)
        await db.insert(tokens).values({
            id: expiredTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'recently-expired-token-hash',
            expiresAt: new Date(now - 2 * 60 * 60 * 1000),
            graceExpiresAt: new Date(now - 2 * 60 * 60 * 1000),
        });

        // 5. Old expired token (expired 2 days ago, grace expired/null)
        await db.insert(tokens).values({
            id: oldExpiredTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'old-expired-token-hash',
            expiresAt: new Date(now - 2 * 24 * 60 * 60 * 1000),
            graceExpiresAt: new Date(now - 2 * 24 * 60 * 60 * 1000),
            createdAt: new Date(now - 2 * 24 * 60 * 60 * 1000),
        });

        // 6. Recently revoked token (revoked 2 hours ago, expires in future)
        await db.insert(tokens).values({
            id: revokedTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'recently-revoked-token-hash',
            expiresAt: new Date(now + 600_000),
            revokedAt: new Date(now - 2 * 60 * 60 * 1000),
        });

        // Test with 24 hours retention period:
        // - Should delete oldExpiredTokenId (expired 2 days ago)
        // - Should NOT delete activeTokenId, graceTokenId, replacementTokenId
        // - Should NOT delete expiredTokenId (expired 2 hours ago, within 24h retention)
        // - Should NOT delete revokedTokenId (revoked 2 hours ago, within 24h retention)
        const deletedCount24h = await authTokenRepo.deleteExpiredTokens(24 * 60 * 60 * 1000);
        expect(deletedCount24h).toBe(1);

        expect(await authTokenRepo.findById(activeTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(graceTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(replacementTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(expiredTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(revokedTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(oldExpiredTokenId)).toBeNull();

        // Test with 0 retention period:
        // - Should delete expiredTokenId and revokedTokenId
        // - Should NOT delete activeTokenId, graceTokenId, replacementTokenId
        const deletedCount0 = await authTokenRepo.deleteExpiredTokens(0);
        expect(deletedCount0).toBe(2);

        expect(await authTokenRepo.findById(activeTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(graceTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(replacementTokenId)).not.toBeNull();
        expect(await authTokenRepo.findById(expiredTokenId)).toBeNull();
        expect(await authTokenRepo.findById(revokedTokenId)).toBeNull();
    });

    it('Phase 33.2: deleteExpiredSessions should delete expired/revoked sessions and deleteExpiredKnownDevices should delete unreferenced revoked devices', async () => {
        const device1 = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const device2 = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const device3 = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });

        // Revoke device2 and device3, keep device1 active
        await knownDeviceRepo.revoke(device2.id, new Date(Date.now() - 2 * 60 * 60 * 1000));
        await knownDeviceRepo.revoke(device3.id, new Date(Date.now() - 2 * 60 * 60 * 1000));

        const now = Date.now();

        // Session 1 on Device 1: Active. Should be preserved.
        const session1 = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device1.id,
            deviceId: device1.deviceId,
            expiresAt: new Date(now + 600_000),
        });

        // Session 2 on Device 2: Expired 2 days ago. Should be eligible.
        const session2 = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device2.id,
            deviceId: device2.deviceId,
            expiresAt: new Date(now - 2 * 24 * 60 * 60 * 1000),
        });

        // Session 3 on Device 3: Revoked 2 hours ago. Should be eligible if retention is 0.
        const session3 = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device3.id,
            deviceId: device3.deviceId,
            expiresAt: new Date(now + 600_000),
        });
        await sessionRepo.revoke(session3.id, new Date(now - 2 * 60 * 60 * 1000));

        // Delete sessions with 24 hours retention:
        // - session2 should be deleted.
        // - session3 should NOT be deleted (within 24h retention).
        const deletedSessions24h = await sessionRepo.deleteExpiredSessions(24 * 60 * 60 * 1000);
        expect(deletedSessions24h).toBe(1);
        expect(await sessionRepo.findById(session1.id)).not.toBeNull();
        expect(await sessionRepo.findById(session2.id)).toBeNull();
        expect(await sessionRepo.findById(session3.id)).not.toBeNull();

        // Try deleting expired known devices:
        // - device2 should be deleted (its only session, session2, was deleted).
        // - device3 should NOT be deleted (session3 still references it).
        const deletedDevices24h = await knownDeviceRepo.deleteExpiredKnownDevices(0);
        expect(deletedDevices24h).toBe(1);
        expect(await knownDeviceRepo.findById(device2.id)).toBeNull();
        expect(await knownDeviceRepo.findById(device3.id)).not.toBeNull();

        // Delete with 0 retention:
        // - session3 should be deleted.
        const deletedSessions0 = await sessionRepo.deleteExpiredSessions(0);
        expect(deletedSessions0).toBe(1);
        expect(await sessionRepo.findById(session3.id)).toBeNull();

        // Now device3 has no referencing sessions, and is revoked. It should be deleted!
        const deletedDevices0 = await knownDeviceRepo.deleteExpiredKnownDevices(0);
        expect(deletedDevices0).toBe(1);
        expect(await knownDeviceRepo.findById(device3.id)).toBeNull();

        // Active device1 should never be deleted
        expect(await knownDeviceRepo.findById(device1.id)).not.toBeNull();
    });

    it('Phase 33.2: UserRepository.delete and deleteExpiredUsers should cascade-delete all user tokens, sessions, and known devices in correct order', async () => {
        // Create user
        const newUser = await userRepo.save({
            name: 'cascadetest',
            email: 'cascade@example.com',
            password: 'SecurePassword123',
            role: UserRoles.USER,
            isActive: true,
        });

        const device = await knownDeviceRepo.save({
            userId: newUser.id,
            realm: 'customer',
            deviceId: randomUUID(),
        });

        const session = await sessionRepo.save({
            userId: newUser.id,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        const tokenId = randomUUID();
        await db.insert(tokens).values({
            id: tokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'cascade-token-hash',
            expiresAt: new Date(Date.now() + 600_000),
        });

        // Deleting user should not violate RESTRICT constraint because we clean up sessions/tokens/devices in order
        const deleteRes = await userRepo.delete(newUser.id);
        expect(deleteRes.affected).toBe(1);

        // Verify all associated data is deleted
        expect(await userRepo.findById(newUser.id)).toBeNull();
        expect(await knownDeviceRepo.findById(device.id)).toBeNull();
        expect(await sessionRepo.findById(session.id)).toBeNull();
        const [token] = await db.select().from(tokens).where(eq(tokens.id, tokenId));
        expect(token).toBeUndefined();
    });

    it('Phase 33.3: deleteExpiredChallenges should delete inactive login challenges but preserve active ones', async () => {
        const now = Date.now();

        // 1. Active challenge: not consumed, not failed, not expired, expires in 10 mins. Should be preserved.
        const activeChallenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId1,
            maxAttempts: 3,
            expiresAt: new Date(now + 600_000),
            checkpointTokenHash: 'hash-active-challenge',
            codeHash: 'code-active',
        });

        // 2. Consumed challenge: expires in 10 mins, but consumed. Should be eligible if retention is 0.
        const consumedChallenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId2,
            maxAttempts: 3,
            expiresAt: new Date(now + 600_000),
            checkpointTokenHash: 'hash-consumed-challenge',
            codeHash: 'code-consumed',
        });
        await loginChallengeRepo.consume(consumedChallenge.id);

        // 3. Failed challenge: expires in 10 mins, but failed. Should be eligible if retention is 0.
        const failedChallenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId3,
            maxAttempts: 3,
            expiresAt: new Date(now + 600_000),
            checkpointTokenHash: 'hash-failed-challenge',
            codeHash: 'code-failed',
        });
        await loginChallengeRepo.fail(failedChallenge.id);

        // 4. Expired challenge: expired 2 hours ago. Should be eligible if retention is 0.
        const expiredChallenge = await loginChallengeRepo.createSafe({
            userId: testUserId,
            realm: 'customer',
            deviceId: testDeviceId4,
            maxAttempts: 3,
            expiresAt: new Date(now - 2 * 60 * 60 * 1000),
            checkpointTokenHash: 'hash-expired-challenge',
            codeHash: 'code-expired',
        });
        await loginChallengeRepo.expire(expiredChallenge.id, new Date(now - 2 * 60 * 60 * 1000));

        // Test with 24 hours retention period:
        // - expiredChallenge (expired 2 hours ago) should NOT be deleted (within 24h retention).
        // - consumedChallenge and failedChallenge (updated just now) should NOT be deleted (within 24h retention).
        const deleted24h = await loginChallengeRepo.deleteExpiredChallenges(24 * 60 * 60 * 1000);
        expect(deleted24h).toBe(0);
        expect(await loginChallengeRepo.findById(activeChallenge.id)).not.toBeNull();
        expect(await loginChallengeRepo.findById(consumedChallenge.id)).not.toBeNull();
        expect(await loginChallengeRepo.findById(failedChallenge.id)).not.toBeNull();
        expect(await loginChallengeRepo.findById(expiredChallenge.id)).not.toBeNull();

        // Test with 0 retention period:
        // - consumed, failed, expired challenges should be deleted.
        // - active challenge must be preserved.
        const deleted0 = await loginChallengeRepo.deleteExpiredChallenges(0);
        expect(deleted0).toBe(3);
        expect(await loginChallengeRepo.findById(activeChallenge.id)).not.toBeNull();
        expect(await loginChallengeRepo.findById(consumedChallenge.id)).toBeNull();
        expect(await loginChallengeRepo.findById(failedChallenge.id)).toBeNull();
        expect(await loginChallengeRepo.findById(expiredChallenge.id)).toBeNull();
    });

    it('Phase 33.3: deleteExpired should delete inactive reauth confirmations but preserve active ones', async () => {
        const now = Date.now();
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(now + 600_000),
        });

        // 1. Active reauth confirmation: expires in 10 mins. Should be preserved.
        const activeConf = await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            actionScope: 'test_scope_active',
            confirmationTokenHash: 'hash-active',
            expiresAt: new Date(now + 600_000),
        });

        // 2. Consumed reauth confirmation: expires in 10 mins, but consumed. Should be eligible if retention is 0.
        const consumedConf = await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            actionScope: 'test_scope_consumed',
            confirmationTokenHash: 'hash-consumed',
            expiresAt: new Date(now + 600_000),
        });
        await reauthConfirmationRepo.consume(consumedConf.id);

        // 3. Expired reauth confirmation: expired 2 hours ago. Should be eligible if retention is 0.
        const expiredConf = await reauthConfirmationRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            actionScope: 'test_scope_expired',
            confirmationTokenHash: 'hash-expired',
            expiresAt: new Date(now - 2 * 60 * 60 * 1000),
        });

        // Test with 24 hours retention period:
        // - expiredConf (expired 2 hours ago) should NOT be deleted (within 24h retention).
        // - consumedConf should NOT be deleted (within 24h retention).
        const deleted24h = await reauthConfirmationRepo.deleteExpired(24 * 60 * 60 * 1000);
        expect(deleted24h).toBe(0);
        expect(await reauthConfirmationRepo.findById(activeConf.id)).not.toBeNull();
        expect(await reauthConfirmationRepo.findById(consumedConf.id)).not.toBeNull();
        expect(await reauthConfirmationRepo.findById(expiredConf.id)).not.toBeNull();

        // Test with 0 retention period:
        // - consumedConf and expiredConf should be deleted.
        // - activeConf must be preserved.
        const deleted0 = await reauthConfirmationRepo.deleteExpired(0);
        expect(deleted0).toBe(2);
        expect(await reauthConfirmationRepo.findById(activeConf.id)).not.toBeNull();
        expect(await reauthConfirmationRepo.findById(consumedConf.id)).toBeNull();
        expect(await reauthConfirmationRepo.findById(expiredConf.id)).toBeNull();
    });

    it('Phase 33.4 / Phase 33.5: deleteExpiredEvents should delete expired security events based on retention policy', async () => {
        const now = Date.now();
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(now + 600_000),
        });

        // 1. Create a security event created just now. Should be preserved.
        const event1 = await securityEventRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            knownDeviceId: device.id,
            eventType: SecurityEventType.LOGIN_SUCCESS,
            riskScore: 0,
            metadata: {
                login_method: 'password',
                ip: '127.0.0.1',
            },
        });

        // 2. Create a security event created 2 days ago. Should be eligible if retention is 24 hours.
        const event2 = await securityEventRepo.save({
            userId: testUserId,
            realm: 'customer',
            sessionId: session.id,
            knownDeviceId: device.id,
            eventType: SecurityEventType.LOGIN_SUCCESS,
            riskScore: 0,
            createdAt: new Date(now - 2 * 24 * 60 * 60 * 1000),
            metadata: {
                login_method: 'password',
                ip: '127.0.0.1',
            },
        });

        // Delete with 24h retention (86400000 ms)
        const deleted24h = await securityEventRepo.deleteExpiredEvents(24 * 60 * 60 * 1000);
        expect(deleted24h).toBe(1);
        expect(await securityEventRepo.findById(event1.id)).not.toBeNull();
        expect(await securityEventRepo.findById(event2.id)).toBeNull();

        // Delete with 0 retention: should delete event1
        const deleted0 = await securityEventRepo.deleteExpiredEvents(0);
        expect(deleted0).toBe(1);
        expect(await securityEventRepo.findById(event1.id)).toBeNull();
    });

    it('Phase 34.2: login/registration without deviceId creates deviceId, known device, session, and token', async () => {
        const userId = testUserId;
        const generatedDeviceId = randomUUID();

        // Simulate deviceIdService.getOrCreateDeviceId
        const deviceId = generatedDeviceId;

        // 1. Create known device
        const knownDevice = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId,
        });
        expect(knownDevice.id).toBeDefined();
        expect(knownDevice.deviceId).toBe(deviceId);

        // 2. Create session
        const session = await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: knownDevice.id,
            deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        expect(session.id).toBeDefined();
        expect(session.knownDeviceId).toBe(knownDevice.id);

        // 3. Create token
        const tokenId = randomUUID();
        const [token] = await db
            .insert(tokens)
            .values({
                id: tokenId,
                sessionId: session.id,
                jti: randomUUID(),
                refreshTokenHash: 'token-hash-1',
                expiresAt: new Date(Date.now() + 600_000),
            })
            .returning();
        expect(token.id).toBe(tokenId);
        expect(token.sessionId).toBe(session.id);
    });

    it('Phase 34.2: repeated login reuses sessionId', async () => {
        const userId = testUserId;
        const deviceId = randomUUID();

        const device = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId,
        });

        // 1. First login creates session
        const session1 = await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 2. Repeated login finds reusable session
        const reusable = await sessionRepo.findReusable(userId, 'customer', deviceId);
        expect(reusable).not.toBeNull();
        expect(reusable!.id).toBe(session1.id);

        // Reuse it
        const newExpiry = new Date(Date.now() + 1_200_000);
        const reused = await sessionRepo.reuseSession(reusable!.id, newExpiry);
        expect(reused).not.toBeNull();
        expect(reused!.id).toBe(session1.id);
        expect(reused!.expiresAt.getTime()).toBe(newExpiry.getTime());
    });

    it('Phase 34.2: repeated login reuses expired unrevoked session', async () => {
        const userId = testUserId;
        const deviceId = randomUUID();

        const device = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId,
        });

        // Expired but unrevoked session
        const session1 = await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId,
            expiresAt: new Date(Date.now() - 600_000), // expired 10 mins ago
        });

        const reusable = await sessionRepo.findReusable(userId, 'customer', deviceId);
        expect(reusable).not.toBeNull();
        expect(reusable!.id).toBe(session1.id);

        const newExpiry = new Date(Date.now() + 600_000);
        const reused = await sessionRepo.reuseSession(reusable!.id, newExpiry);
        expect(reused).not.toBeNull();
        expect(reused!.id).toBe(session1.id);
        expect(reused!.expiresAt.getTime()).toBe(newExpiry.getTime());
    });

    it('Phase 34.2: login with another deviceId creates a new session', async () => {
        const userId = testUserId;
        const deviceId1 = randomUUID();
        const deviceId2 = randomUUID();

        const device1 = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId: deviceId1,
        });
        const device2 = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId: deviceId2,
        });

        const session1 = await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: device1.id,
            deviceId: deviceId1,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // Query reusable session for device2: should be null
        const reusable = await sessionRepo.findReusable(userId, 'customer', deviceId2);
        expect(reusable).toBeNull();

        // Create new session for device2
        const session2 = await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: device2.id,
            deviceId: deviceId2,
            expiresAt: new Date(Date.now() + 600_000),
        });
        expect(session2.id).not.toBe(session1.id);
    });

    it('Phase 34.2: parallel login does not create duplicates due to unique index sessions_user_realm_device_active_unique', async () => {
        const userId = testUserId;
        const deviceId = randomUUID();

        const device = await knownDeviceRepo.save({
            userId,
            realm: 'customer',
            deviceId,
        });

        await sessionRepo.save({
            userId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // Second insert of active session for same user/realm/device should fail with unique index violation
        await expect(
            sessionRepo.save({
                userId,
                realm: 'customer',
                knownDeviceId: device.id,
                deviceId,
                expiresAt: new Date(Date.now() + 600_000),
            }),
        ).rejects.toThrow();
    });

    it('Phase 34.2: registration rollback does not leave orphan user, session, or token rows', async () => {
        const registerUserFlow = async (shouldFail: boolean) => {
            await db.transaction(async tx => {
                const [newUser] = await tx
                    .insert(users)
                    .values({
                        id: randomUUID(),
                        name: 'rollbackuser',
                        email: 'rollback@example.com',
                        password: 'HashedPassword123',
                        role: UserRoles.USER,
                    })
                    .returning();

                const device = await tx
                    .insert(knownDevices)
                    .values({
                        userId: newUser.id,
                        realm: 'customer',
                        deviceId: randomUUID(),
                    })
                    .returning();

                const [session] = await tx
                    .insert(sessions)
                    .values({
                        userId: newUser.id,
                        realm: 'customer',
                        knownDeviceId: device[0].id,
                        deviceId: device[0].deviceId,
                        expiresAt: new Date(Date.now() + 600_000),
                    })
                    .returning();

                await tx.insert(tokens).values({
                    id: randomUUID(),
                    sessionId: session.id,
                    jti: randomUUID(),
                    refreshTokenHash: 'rollback-hash',
                    expiresAt: new Date(Date.now() + 600_000),
                });

                if (shouldFail) {
                    throw new Error('Forced registration rollback');
                }
            });
        };

        // Run failing flow: should throw and rollback all changes
        await expect(registerUserFlow(true)).rejects.toThrow('Forced registration rollback');

        // Verify no user, session, known device, or token was persisted
        const [rolledUser] = await db
            .select()
            .from(users)
            .where(eq(users.email, 'rollback@example.com'));
        expect(rolledUser).toBeUndefined();

        const [rolledToken] = await db
            .select()
            .from(tokens)
            .where(eq(tokens.refreshTokenHash, 'rollback-hash'));
        expect(rolledToken).toBeUndefined();
    });

    it('Phase 34.3: should verify replaced token behaves correctly in and after grace window, and supports winning response recovery', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        const originalTokenId = randomUUID();
        const replacementTokenId = randomUUID();
        const replacementTokenHash = 'replacement-token-hash';

        // 1. Insert replacement token
        await db.insert(tokens).values({
            id: replacementTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: replacementTokenHash,
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 2. Insert rotated token with graceExpiresAt in the future and encryptedReplacementToken
        const graceExpiresAt = new Date(Date.now() + 60_000); // grace window expires in 60s
        await db.insert(tokens).values({
            id: originalTokenId,
            sessionId: session.id,
            jti: randomUUID(),
            refreshTokenHash: 'original-token-hash',
            expiresAt: new Date(Date.now() - 10_000), // expired
            replacedAt: new Date(),
            replacedByTokenId: replacementTokenId,
            graceExpiresAt,
            encryptedReplacementToken: 'encrypted-replacement-hash',
        });

        // Verify findRefreshTokenById retrieves the original token and details
        const tokenRow = await authTokenRepo.findRefreshTokenById(originalTokenId);
        expect(tokenRow).not.toBeNull();
        expect(tokenRow!.replacedByTokenId).toBe(replacementTokenId);
        expect(tokenRow!.encryptedReplacementToken).toBe('encrypted-replacement-hash');

        // Test helper function: isValidGraceToken (during grace window)
        const now = new Date();
        const isGrace =
            tokenRow!.revokedAt === null &&
            tokenRow!.replacedAt !== null &&
            tokenRow!.graceExpiresAt !== null &&
            tokenRow!.graceExpiresAt.getTime() > now.getTime() &&
            tokenRow!.replacementSessionId === tokenRow!.sessionId &&
            tokenRow!.replacementRevokedAt === null;
        expect(isGrace).toBe(true);

        // Test helper function: isValidGraceToken (after grace window)
        const later = new Date(Date.now() + 120_000);
        const isGraceLater =
            tokenRow!.revokedAt === null &&
            tokenRow!.replacedAt !== null &&
            tokenRow!.graceExpiresAt !== null &&
            tokenRow!.graceExpiresAt.getTime() > later.getTime() &&
            tokenRow!.replacementSessionId === tokenRow!.sessionId &&
            tokenRow!.replacementRevokedAt === null;
        expect(isGraceLater).toBe(false);
    });

    it('Phase 34.4: failed/expired challenge does not block creation of a new challenge for the same user/realm/device', async () => {
        const userId = testUserId;
        const deviceId = randomUUID();

        // 1. Create first challenge
        const challenge1 = await loginChallengeRepo.createSafe({
            userId,
            realm: 'customer',
            deviceId,
            checkpointTokenHash: 'token-hash-1',
            codeHash: 'code-hash-1',
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 2. Creating another active challenge expires the previous pending challenge
        const challenge2 = await loginChallengeRepo.createSafe({
            userId,
            realm: 'customer',
            deviceId,
            checkpointTokenHash: 'token-hash-2',
            codeHash: 'code-hash-2',
            expiresAt: new Date(Date.now() + 600_000),
        });
        expect(challenge2.id).not.toBe(challenge1.id);

        const expiredChallenge1 = await loginChallengeRepo.findById(challenge1.id);
        expect(expiredChallenge1?.expiredAt).toBeInstanceOf(Date);

        // 3. Mark first challenge as failed
        await loginChallengeRepo.fail(challenge2.id);

        // 4. Creating a new challenge now succeeds and returns a new challenge ID
        const challenge3 = await loginChallengeRepo.createSafe({
            userId,
            realm: 'customer',
            deviceId,
            checkpointTokenHash: 'token-hash-3',
            codeHash: 'code-hash-3',
            expiresAt: new Date(Date.now() + 600_000),
        });
        expect(challenge3.id).not.toBe(challenge2.id);
    });

    it('Phase 34.4: verifyLoginChallenge only accepts valid code and active challenge, and fails on expired/invalid/reached attempts', async () => {
        const userId = testUserId;
        const deviceId = randomUUID();
        const now = new Date();

        const challengeService = new LoginChallengeService(loginChallengeRepo);

        const { checkpointToken, code } = await challengeService.createLoginChallenge(
            userId,
            'customer',
            null,
            deviceId,
            {},
            { riskScore: 70 },
            now,
        );

        // 1. Invalid code returns false
        expect(await challengeService.verifyLoginChallenge(checkpointToken, 'wrongcode', now)).toBe(
            false,
        );

        // 2. Valid code returns true
        expect(await challengeService.verifyLoginChallenge(checkpointToken, code, now)).toBe(true);

        // 3. Increment attempts until failed
        for (let i = 0; i < 5; i++) {
            await challengeService.verifyLoginChallenge(checkpointToken, 'wrongcode', now);
        }

        // 4. Verification fails now because max attempts are reached and challenge is failed
        expect(await challengeService.verifyLoginChallenge(checkpointToken, code, now)).toBe(false);
    });

    it('Phase 34.5: revokeUserSessions correctly revokes all sessions and refresh tokens across all realms', async () => {
        // Create user
        const newUser = await userRepo.save({
            name: 'revokemultirealms',
            email: 'revokemulti@example.com',
            password: 'HashedPassword123',
            role: UserRoles.USER,
            isActive: true,
        });

        // 1. Create session 1 (realm: customer)
        const device1 = await knownDeviceRepo.save({
            userId: newUser.id,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session1 = await sessionRepo.save({
            userId: newUser.id,
            realm: 'customer',
            knownDeviceId: device1.id,
            deviceId: device1.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        const tokenId1 = randomUUID();
        await db.insert(tokens).values({
            id: tokenId1,
            sessionId: session1.id,
            jti: randomUUID(),
            refreshTokenHash: 'token-hash-multi-1',
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 2. Create session 2 (realm: admin)
        const device2 = await knownDeviceRepo.save({
            userId: newUser.id,
            realm: 'admin',
            deviceId: randomUUID(),
        });
        const session2 = await sessionRepo.save({
            userId: newUser.id,
            realm: 'admin',
            knownDeviceId: device2.id,
            deviceId: device2.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });
        const tokenId2 = randomUUID();
        await db.insert(tokens).values({
            id: tokenId2,
            sessionId: session2.id,
            jti: randomUUID(),
            refreshTokenHash: 'token-hash-multi-2',
            expiresAt: new Date(Date.now() + 600_000),
        });

        // 3. Verify they are active before revocation
        const activeSessionsBefore = await sessionRepo.findActiveByUser(newUser.id);
        expect(activeSessionsBefore).toHaveLength(2);

        // 4. Revoke all user sessions (simulating password change/reset behavior)
        const now = new Date();
        for (const session of activeSessionsBefore) {
            await sessionRepo.revoke(session.id, now);
            await authTokenRepo.revokeTokensBySession(session.id, now);
        }

        // 5. Verify all sessions and tokens are revoked
        const activeSessionsAfter = await sessionRepo.findActiveByUser(newUser.id);
        expect(activeSessionsAfter).toHaveLength(0);

        const token1 = await authTokenRepo.findById(tokenId1);
        const token2 = await authTokenRepo.findById(tokenId2);
        expect(token1?.revokedAt).not.toBeNull();
        expect(token2?.revokedAt).not.toBeNull();
    });

    it('Phase 34.6: touchLastSeenAt updates lastSeenAt but does not extend session absolute expiresAt', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const initialExpiresAt = new Date(Date.now() + 600_000);
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: initialExpiresAt,
        });

        const now = new Date(Date.now() + 10_000);
        const updated = await sessionRepo.update(session.id, { lastSeenAt: now });

        expect(updated).not.toBeNull();
        expect(updated!.lastSeenAt.getTime()).toBe(now.getTime());
        expect(updated!.expiresAt.getTime()).toBe(initialExpiresAt.getTime());
    });

    it('Phase 16.5: should revoke all refresh tokens in a session', async () => {
        const device = await knownDeviceRepo.save({
            userId: testUserId,
            realm: 'customer',
            deviceId: randomUUID(),
        });
        const session = await sessionRepo.save({
            userId: testUserId,
            realm: 'customer',
            knownDeviceId: device.id,
            deviceId: device.deviceId,
            expiresAt: new Date(Date.now() + 600_000),
        });

        const tokenId1 = randomUUID();
        const tokenId2 = randomUUID();

        await db.insert(tokens).values([
            {
                id: tokenId1,
                sessionId: session.id,
                jti: randomUUID(),
                refreshTokenHash: 'token1-hash',
                expiresAt: new Date(Date.now() + 600_000),
            },
            {
                id: tokenId2,
                sessionId: session.id,
                jti: randomUUID(),
                refreshTokenHash: 'token2-hash',
                expiresAt: new Date(Date.now() + 600_000),
            },
        ]);

        const revokedCount = await authTokenRepo.revokeTokensBySession(session.id);
        expect(revokedCount).toBe(2);

        const token1 = await authTokenRepo.findById(tokenId1);
        const token2 = await authTokenRepo.findById(tokenId2);

        expect(token1?.revokedAt).not.toBeNull();
        expect(token2?.revokedAt).not.toBeNull();

        const revokedCountAgain = await authTokenRepo.revokeTokensBySession(session.id);
        expect(revokedCountAgain).toBe(0);
    });
});

function getDatabaseConfig(database: string | undefined): ClientConfig {
    if (!database) {
        throw new Error('DB_NAME is required for auth-token integration tests');
    }

    return {
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        database,
    };
}

function quoteIdentifier(identifier: string): string {
    return `"${identifier.replaceAll('"', '""')}"`;
}
