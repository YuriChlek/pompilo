CREATE TYPE "public"."identity_outbox_event_status_enum" AS ENUM('pending', 'published', 'failed');--> statement-breakpoint
CREATE TYPE "public"."mail_outbox_status_enum" AS ENUM('pending', 'queued', 'sent', 'failed');--> statement-breakpoint
CREATE TYPE "public"."tenant_role_enum" AS ENUM('OWNER', 'ADMIN', 'MEMBER');--> statement-breakpoint
CREATE TYPE "public"."tenant_status_enum" AS ENUM('ACTIVE', 'SUSPENDED');--> statement-breakpoint
CREATE TYPE "public"."account_status_enum" AS ENUM('ACTIVE', 'DEACTIVATED', 'PENDING_DELETION', 'DELETED');--> statement-breakpoint
CREATE TYPE "public"."identity_user_role_enum" AS ENUM('user', 'platformAdmin', 'superAdmin');--> statement-breakpoint
CREATE TABLE "email_change_challenges" (
	"email_change_challenge_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"new_email" varchar(255) NOT NULL,
	"code_digest" text NOT NULL,
	"attempt_count" integer DEFAULT 0 NOT NULL,
	"locked_until" timestamp with time zone,
	"expires_at" timestamp with time zone NOT NULL,
	"used_at" timestamp with time zone,
	"invalidated_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "password_reset_challenges" (
	"password_reset_challenge_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"selector" varchar(64) NOT NULL,
	"verifier_digest" text NOT NULL,
	"attempt_count" integer DEFAULT 0 NOT NULL,
	"locked_until" timestamp with time zone,
	"expires_at" timestamp with time zone NOT NULL,
	"used_at" timestamp with time zone,
	"invalidated_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "password_reset_challenges_selector_unique" UNIQUE("selector")
);
--> statement-breakpoint
CREATE TABLE "user_settings" (
	"user_id" uuid PRIMARY KEY NOT NULL,
	"timezone" varchar(100) DEFAULT 'UTC' NOT NULL,
	"notify_security_alerts" boolean DEFAULT true NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "email_verifications" (
	"email_verification_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"token_hash" text NOT NULL,
	"expires_at" timestamp with time zone NOT NULL,
	"consumed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "known_devices" (
	"known_device_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"realm" varchar(255) NOT NULL,
	"device_id" uuid NOT NULL,
	"trusted_at" timestamp with time zone,
	"trust_expires_at" timestamp with time zone,
	"revoked_at" timestamp with time zone,
	"first_seen_at" timestamp with time zone DEFAULT now() NOT NULL,
	"last_seen_at" timestamp with time zone DEFAULT now() NOT NULL,
	"last_ip_address" varchar(45),
	"last_country" varchar(2),
	"last_region" varchar(128),
	"last_city" varchar(128),
	"last_user_agent" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "known_devices_realm_check" CHECK ("known_devices"."realm" in ('customer', 'admin'))
);
--> statement-breakpoint
CREATE TABLE "login_challenges" (
	"login_challenge_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"realm" varchar(255) NOT NULL,
	"known_device_id" uuid,
	"device_id" uuid NOT NULL,
	"challenge_type" varchar(255) DEFAULT 'email_code' NOT NULL,
	"checkpoint_token_hash" text NOT NULL,
	"code_hash" text NOT NULL,
	"attempt_count" integer DEFAULT 0 NOT NULL,
	"max_attempts" integer DEFAULT 5 NOT NULL,
	"expires_at" timestamp with time zone NOT NULL,
	"approved_at" timestamp with time zone,
	"consumed_at" timestamp with time zone,
	"failed_at" timestamp with time zone,
	"expired_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"ip_address" varchar(45),
	"country" varchar(2),
	"region" varchar(128),
	"city" varchar(128),
	"user_agent" text,
	"risk_score" integer DEFAULT 0 NOT NULL,
	"risk_reason" text,
	CONSTRAINT "login_challenges_realm_check" CHECK ("login_challenges"."realm" in ('customer', 'admin')),
	CONSTRAINT "login_challenges_attempts_check" CHECK ("login_challenges"."attempt_count" >= 0 and "login_challenges"."max_attempts" > 0 and "login_challenges"."attempt_count" <= "login_challenges"."max_attempts")
);
--> statement-breakpoint
CREATE TABLE "reauth_confirmations" (
	"reauth_confirmation_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"realm" varchar(255) NOT NULL,
	"session_id" uuid NOT NULL,
	"action_scope" varchar(255) NOT NULL,
	"confirmation_token_hash" text NOT NULL,
	"expires_at" timestamp with time zone NOT NULL,
	"consumed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "reauth_confirmations_realm_check" CHECK ("reauth_confirmations"."realm" in ('customer', 'admin'))
);
--> statement-breakpoint
CREATE TABLE "security_events" (
	"security_event_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid,
	"realm" varchar(255) NOT NULL,
	"session_id" uuid,
	"known_device_id" uuid,
	"event_type" varchar(255) NOT NULL,
	"risk_score" integer DEFAULT 0 NOT NULL,
	"risk_reason" text,
	"ip_address" varchar(45),
	"country" varchar(2),
	"region" varchar(128),
	"city" varchar(128),
	"user_agent" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"metadata" jsonb,
	CONSTRAINT "security_events_realm_check" CHECK ("security_events"."realm" in ('customer', 'admin'))
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"session_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"realm" varchar(255) NOT NULL,
	"known_device_id" uuid NOT NULL,
	"device_id" uuid NOT NULL,
	"ip_address" varchar(45),
	"user_agent" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"last_seen_at" timestamp with time zone DEFAULT now() NOT NULL,
	"expires_at" timestamp with time zone NOT NULL,
	"revoked_at" timestamp with time zone,
	"last_country" varchar(2),
	"last_region" varchar(128),
	"last_city" varchar(128),
	"risk_score" integer DEFAULT 0 NOT NULL,
	"risk_reason" text,
	CONSTRAINT "sessions_realm_check" CHECK ("sessions"."realm" in ('customer', 'admin')),
	CONSTRAINT "sessions_risk_score_check" CHECK ("sessions"."risk_score" >= 0)
);
--> statement-breakpoint
CREATE TABLE "tokens" (
	"token_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" uuid NOT NULL,
	"jti" varchar NOT NULL,
	"refresh_token_hash" text NOT NULL,
	"encrypted_replacement_token" text,
	"expires_at" timestamp with time zone NOT NULL,
	"revoked_at" timestamp with time zone,
	"replaced_by_token_id" uuid,
	"replaced_at" timestamp with time zone,
	"grace_expires_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "data_patches" (
	"patch_name" text PRIMARY KEY NOT NULL,
	"checksum" text NOT NULL,
	"applied_at" timestamp with time zone DEFAULT now() NOT NULL,
	"duration_ms" integer NOT NULL,
	"description" text,
	"app_version" text,
	"node_env" text
);
--> statement-breakpoint
CREATE TABLE "identity_outbox_events" (
	"identity_outbox_event_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"event_id" uuid NOT NULL,
	"event_type" varchar(128) NOT NULL,
	"event_version" integer DEFAULT 1 NOT NULL,
	"aggregate_id" uuid NOT NULL,
	"tenant_id" uuid NOT NULL,
	"idempotency_key" varchar(255) NOT NULL,
	"status" "identity_outbox_event_status_enum" DEFAULT 'pending' NOT NULL,
	"payload" jsonb NOT NULL,
	"available_at" timestamp with time zone DEFAULT now() NOT NULL,
	"published_at" timestamp with time zone,
	"last_error" varchar(1024),
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "mail_audit_events" (
	"mail_audit_event_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"action" varchar(100) NOT NULL,
	"admin_user_id" uuid NOT NULL,
	"payload" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "mail_outbox" (
	"mail_outbox_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"idempotency_key" varchar(255) NOT NULL,
	"status" "mail_outbox_status_enum" DEFAULT 'pending' NOT NULL,
	"payload_encrypted" text,
	"payload_json" jsonb,
	"priority" integer DEFAULT 0 NOT NULL,
	"attempt_count" integer DEFAULT 0 NOT NULL,
	"last_error" text,
	"available_at" timestamp with time zone DEFAULT now() NOT NULL,
	"locked_at" timestamp with time zone,
	"locked_by" varchar(255),
	"queued_job_id" varchar(255),
	"queued_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "mail_outbox_payload_exactly_one_check" CHECK ((payload_encrypted IS NOT NULL) <> (payload_json IS NOT NULL))
);
--> statement-breakpoint
CREATE TABLE "mail_settings" (
	"mail_settings_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"singleton_key" boolean DEFAULT true NOT NULL,
	"provider" varchar(32) DEFAULT 'smtp' NOT NULL,
	"smtp_host" varchar(255) NOT NULL,
	"smtp_port" integer NOT NULL,
	"smtp_secure" boolean DEFAULT false NOT NULL,
	"smtp_user" varchar(255),
	"smtp_password_encrypted" text,
	"from_address" varchar(255) NOT NULL,
	"from_name" varchar(255) NOT NULL,
	"reply_to" varchar(255),
	"client_public_url" varchar(255),
	"enabled" boolean DEFAULT true NOT NULL,
	"last_verified_at" timestamp with time zone,
	"last_verification_error" text,
	"confirmed_at" timestamp with time zone,
	"confirmed_by_user_id" uuid,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_by_user_id" uuid,
	CONSTRAINT "mail_settings_singleton_key_unique" UNIQUE("singleton_key")
);
--> statement-breakpoint
CREATE TABLE "memberships" (
	"user_id" uuid NOT NULL,
	"tenant_id" uuid NOT NULL,
	"role" "tenant_role_enum" DEFAULT 'MEMBER' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "memberships_user_id_tenant_id_pk" PRIMARY KEY("user_id","tenant_id")
);
--> statement-breakpoint
CREATE TABLE "tenants" (
	"tenant_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar(255) NOT NULL,
	"status" "tenant_status_enum" DEFAULT 'ACTIVE' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "users" (
	"user_id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar(255) NOT NULL,
	"email" varchar(255) NOT NULL,
	"password" text NOT NULL,
	"role" "identity_user_role_enum" DEFAULT 'user' NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"account_status" "account_status_enum" DEFAULT 'ACTIVE' NOT NULL,
	"email_verified_at" timestamp with time zone,
	"pending_email_change" varchar(255),
	"deletion_scheduled_at" timestamp with time zone,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "email_change_challenges" ADD CONSTRAINT "email_change_challenges_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "password_reset_challenges" ADD CONSTRAINT "password_reset_challenges_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_settings" ADD CONSTRAINT "user_settings_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "email_verifications" ADD CONSTRAINT "email_verifications_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "known_devices" ADD CONSTRAINT "known_devices_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "login_challenges" ADD CONSTRAINT "login_challenges_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "login_challenges" ADD CONSTRAINT "login_challenges_known_device_id_known_devices_known_device_id_fk" FOREIGN KEY ("known_device_id") REFERENCES "public"."known_devices"("known_device_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "reauth_confirmations" ADD CONSTRAINT "reauth_confirmations_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "reauth_confirmations" ADD CONSTRAINT "reauth_confirmations_session_id_sessions_session_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."sessions"("session_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "security_events" ADD CONSTRAINT "security_events_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "security_events" ADD CONSTRAINT "security_events_session_id_sessions_session_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."sessions"("session_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "security_events" ADD CONSTRAINT "security_events_known_device_id_known_devices_known_device_id_fk" FOREIGN KEY ("known_device_id") REFERENCES "public"."known_devices"("known_device_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_known_device_id_known_devices_known_device_id_fk" FOREIGN KEY ("known_device_id") REFERENCES "public"."known_devices"("known_device_id") ON DELETE restrict ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tokens" ADD CONSTRAINT "tokens_session_id_sessions_session_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."sessions"("session_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tokens" ADD CONSTRAINT "tokens_replaced_by_token_id_tokens_token_id_fk" FOREIGN KEY ("replaced_by_token_id") REFERENCES "public"."tokens"("token_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mail_audit_events" ADD CONSTRAINT "mail_audit_events_admin_user_id_users_user_id_fk" FOREIGN KEY ("admin_user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mail_settings" ADD CONSTRAINT "mail_settings_confirmed_by_user_id_users_user_id_fk" FOREIGN KEY ("confirmed_by_user_id") REFERENCES "public"."users"("user_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mail_settings" ADD CONSTRAINT "mail_settings_updated_by_user_id_users_user_id_fk" FOREIGN KEY ("updated_by_user_id") REFERENCES "public"."users"("user_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "memberships" ADD CONSTRAINT "memberships_user_id_users_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "memberships" ADD CONSTRAINT "memberships_tenant_id_tenants_tenant_id_fk" FOREIGN KEY ("tenant_id") REFERENCES "public"."tenants"("tenant_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE UNIQUE INDEX "active_email_change_challenge_per_user_idx" ON "email_change_challenges" USING btree ("user_id") WHERE used_at IS NULL AND invalidated_at IS NULL;--> statement-breakpoint
CREATE INDEX "email_change_user_id_idx" ON "email_change_challenges" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "email_change_expires_at_idx" ON "email_change_challenges" USING btree ("expires_at");--> statement-breakpoint
CREATE INDEX "email_change_composite_idx" ON "email_change_challenges" USING btree ("user_id","used_at","expires_at");--> statement-breakpoint
CREATE UNIQUE INDEX "active_password_reset_challenge_per_user_idx" ON "password_reset_challenges" USING btree ("user_id") WHERE used_at IS NULL AND invalidated_at IS NULL;--> statement-breakpoint
CREATE INDEX "password_reset_user_id_idx" ON "password_reset_challenges" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "password_reset_expires_at_idx" ON "password_reset_challenges" USING btree ("expires_at");--> statement-breakpoint
CREATE INDEX "password_reset_composite_idx" ON "password_reset_challenges" USING btree ("user_id","used_at","expires_at");--> statement-breakpoint
CREATE UNIQUE INDEX "email_verifications_token_hash_unique" ON "email_verifications" USING btree ("token_hash");--> statement-breakpoint
CREATE INDEX "email_verifications_expiry_idx" ON "email_verifications" USING btree ("expires_at") WHERE consumed_at is null;--> statement-breakpoint
CREATE UNIQUE INDEX "known_devices_user_realm_device_unique" ON "known_devices" USING btree ("user_id","realm","device_id") WHERE revoked_at is null;--> statement-breakpoint
CREATE INDEX "known_devices_user_last_seen_idx" ON "known_devices" USING btree ("user_id","realm","last_seen_at") WHERE revoked_at is null;--> statement-breakpoint
CREATE UNIQUE INDEX "login_challenges_active_unique" ON "login_challenges" USING btree ("user_id","realm","device_id") WHERE consumed_at is null and failed_at is null and expired_at is null;--> statement-breakpoint
CREATE INDEX "login_challenges_expiry_idx" ON "login_challenges" USING btree ("expires_at");--> statement-breakpoint
CREATE UNIQUE INDEX "reauth_confirmations_token_hash_unique" ON "reauth_confirmations" USING btree ("confirmation_token_hash");--> statement-breakpoint
CREATE INDEX "reauth_confirmations_expiry_idx" ON "reauth_confirmations" USING btree ("expires_at") WHERE consumed_at is null;--> statement-breakpoint
CREATE INDEX "security_events_device_idx" ON "security_events" USING btree ("known_device_id") WHERE known_device_id is not null;--> statement-breakpoint
CREATE INDEX "security_events_user_occurred_idx" ON "security_events" USING btree ("user_id","realm","created_at");--> statement-breakpoint
CREATE INDEX "sessions_user_active_idx" ON "sessions" USING btree ("user_id","realm") WHERE revoked_at is null;--> statement-breakpoint
CREATE INDEX "sessions_last_seen_idx" ON "sessions" USING btree ("user_id","realm","last_seen_at") WHERE revoked_at is null;--> statement-breakpoint
CREATE INDEX "sessions_expiry_idx" ON "sessions" USING btree ("expires_at") WHERE revoked_at is null;--> statement-breakpoint
CREATE UNIQUE INDEX "sessions_user_realm_device_active_unique" ON "sessions" USING btree ("user_id","realm","device_id") WHERE revoked_at is null;--> statement-breakpoint
CREATE UNIQUE INDEX "tokens_jti_unique_idx" ON "tokens" USING btree ("jti");--> statement-breakpoint
CREATE INDEX "tokens_session_valid_idx" ON "tokens" USING btree ("session_id","expires_at") WHERE revoked_at is null;--> statement-breakpoint
CREATE INDEX "tokens_grace_idx" ON "tokens" USING btree ("grace_expires_at") WHERE replaced_at is not null and revoked_at is null;--> statement-breakpoint
CREATE UNIQUE INDEX "identity_outbox_event_id_idx" ON "identity_outbox_events" USING btree ("event_id");--> statement-breakpoint
CREATE UNIQUE INDEX "identity_outbox_idempotency_key_idx" ON "identity_outbox_events" USING btree ("idempotency_key");--> statement-breakpoint
CREATE INDEX "identity_outbox_status_idx" ON "identity_outbox_events" USING btree ("status");--> statement-breakpoint
CREATE INDEX "identity_outbox_available_idx" ON "identity_outbox_events" USING btree ("status","available_at");--> statement-breakpoint
CREATE INDEX "identity_outbox_tenant_idx" ON "identity_outbox_events" USING btree ("tenant_id");--> statement-breakpoint
CREATE INDEX "mail_audit_events_action_created_at_idx" ON "mail_audit_events" USING btree ("action","created_at" DESC NULLS LAST);--> statement-breakpoint
CREATE INDEX "mail_audit_events_admin_user_id_created_at_idx" ON "mail_audit_events" USING btree ("admin_user_id","created_at" DESC NULLS LAST);--> statement-breakpoint
CREATE UNIQUE INDEX "mail_outbox_idempotency_key_idx" ON "mail_outbox" USING btree ("idempotency_key");--> statement-breakpoint
CREATE INDEX "mail_outbox_status_idx" ON "mail_outbox" USING btree ("status");--> statement-breakpoint
CREATE INDEX "mail_outbox_claim_idx" ON "mail_outbox" USING btree ("priority" DESC NULLS LAST,"available_at") WHERE status = 'pending';--> statement-breakpoint
CREATE INDEX "mail_outbox_stale_reclaim_idx" ON "mail_outbox" USING btree ("locked_at") WHERE status = 'pending' AND locked_at IS NOT NULL;--> statement-breakpoint
CREATE UNIQUE INDEX "users_name_unique_idx" ON "users" USING btree ("name");--> statement-breakpoint
CREATE UNIQUE INDEX "users_email_unique_idx" ON "users" USING btree ("email");--> statement-breakpoint
CREATE UNIQUE INDEX "users_normalized_email_unique_idx" ON "users" USING btree (lower("email"));
--> statement-breakpoint
CREATE OR REPLACE FUNCTION notify_mail_outbox_inserted()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('mail_outbox_inserted', NEW.mail_outbox_id::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
--> statement-breakpoint
DROP TRIGGER IF EXISTS mail_outbox_inserted_trigger ON mail_outbox;
--> statement-breakpoint
CREATE TRIGGER mail_outbox_inserted_trigger
AFTER INSERT ON mail_outbox
FOR EACH ROW
EXECUTE FUNCTION notify_mail_outbox_inserted();
--> statement-breakpoint
ALTER TABLE "mail_outbox" SET (
	autovacuum_vacuum_scale_factor = 0.05,
	autovacuum_vacuum_threshold = 100
);
