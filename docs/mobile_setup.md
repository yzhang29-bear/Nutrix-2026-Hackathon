# Mobile app setup (Clerk sign-in)

The Expo app in `mobile/` uses Clerk for email/password authentication and has three exclusive portal roles:

- `Patient` — public, self-service email/password registration.
- `Clinician` — invite-only account created by the Nutrix team.
- `Admin` — invite-only account created by the Nutrix team.

The Admin / Clinician / Patient selector is a portal check, not a way to choose access. An account that signs in through a different portal is immediately signed out and shown an authorization error.

## Mobile setup

1. In the Clerk Dashboard, enable **Native API** under **Native applications**.
2. Under **User & authentication**, enable **Email address**, **Password**, email-code verification, and first-name collection. First name is Nutrix's required display name.
3. Create `mobile/.env` locally (it is ignored by Git):

   ```env
   EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
   ```

4. Start the app:

   ```bash
   cd mobile
   npm ci
   npx expo start
   ```

Only the Clerk publishable key belongs in the Expo app. Never put `CLERK_SECRET_KEY` in a mobile build, an `.env` file that is committed, or any client-side code.

## Provision an Admin or Clinician

Before giving a staff member access:

1. Create or invite their Clerk user from a trusted team process.
2. Set `publicMetadata.role` to exactly `admin` or `clinician` using the Clerk Dashboard/Backend API. Never allow the mobile client to write this value.
3. Verify the user can only enter through the matching Nutrix portal tab.
4. Configure Clerk's session-token customization to add the role claim used by the AWS API. The AWS service must verify the Clerk JWT and authorize its role claim; it must not trust a role sent in a request body.

Publicly registered users receive `publicMetadata.role: "patient"` through the trusted Clerk webhook in `services/clerk-role-webhook/`. The mobile app never writes public metadata. This makes the patient role visible in the Clerk Dashboard while preventing self-assignment of staff roles.

## Patient role webhook

Run `npm install && npm run build` in `services/clerk-role-webhook/`, then deploy its `dist/` directory with the production dependencies as a Node.js 20+ AWS Lambda. Set the Lambda handler to `index.handler` and expose it through a **Lambda Function URL** (public `POST` access is required for Clerk delivery). Configure these Lambda environment variables from secure secret storage, never from the Expo app:

- `CLERK_SECRET_KEY` — Clerk Backend API secret key.
- `CLERK_WEBHOOK_SIGNING_SECRET` — signing secret for this specific Clerk webhook endpoint.

In Clerk Dashboard, create a webhook endpoint using the Lambda Function URL and subscribe it to `user.created`. The handler verifies the Clerk signature, then writes `publicMetadata.role: "patient"` with Clerk's Backend API. It preserves a pre-existing `admin` or `clinician` role created by a trusted backend.

Webhook delivery is asynchronous. A just-created patient's role may appear in the Clerk Dashboard a moment after signup; the mobile app safely resolves an absent role as Patient during that brief interval.

For a staff account created manually in the Dashboard, the webhook initially assigns `patient`. Before sending the invitation, overwrite the role with exactly `admin` or `clinician`; the mobile app and future AWS API then treat that staff role as authoritative.

## Calling the prediction API

The app sends the Clerk session token as `Authorization: Bearer <token>` to the `/v1/mobile/*` routes. The AWS service verifies the token's signature, issuer, audience, expiry and role claim before it reads any data or runs the model; a user can only reach their own data. Contract: [mobile_gateway_contract.md](mobile_gateway_contract.md).
