# Mobile gateway contract

The Expo app calls this gateway with a Clerk session token. The gateway verifies its signature against the configured
Clerk issuer JWKS, plus `iss`, `aud`, expiry, and Nutrix role claim. For the demo, it reads the signed
`pmrf_person_id` claim and permits only IDs listed in `MOBILE_DEMO_PERSON_IDS`; the app cannot choose a patient ID.
It injects the PMRF `x-api-key` only inside the service process. The PMRF key is never supplied by, or exposed to,
the mobile client.

Every endpoint requires `Authorization: Bearer <Clerk session token>`.

| Method | Path | Response / behavior |
| --- | --- | --- |
| `GET` | `/v1/mobile/readiness` | `{ status, daysCollected, requiredDays }`. `status` is `collecting_baseline`, `processing_baseline`, `needs_recent_cgm`, or `ready`. |
| `POST` | `/v1/mobile/photos/upload-url` | Proxies PMRF's upload-url call and returns `{ photoKey, uploadUrl, headers }`. The app uploads the normalized JPEG directly to `uploadUrl` with the returned headers. |
| `POST` | `/v1/mobile/predictions` | Body: `{ photoKey, mealTime, mealType, logMeal }`; proxies PMRF `/predict` and returns its prediction JSON unchanged. |

The gateway translates PMRF errors to the same status codes and returns `{ "detail": "human-readable reason" }`.
The mobile app treats `401` as an expired session, `404` as an unprovisioned profile, `409` as baseline/CGM not ready,
`413` as an image-size problem, and `422` as invalid request data.

Before deployment, configure Clerk's session-token customization with `aud` set to `CLERK_JWT_AUDIENCE` and a `role`
claim containing `patient`, `clinician`, or `admin`. Pass the two non-secret values, plus `CLERK_JWT_ISSUER`, to
`scripts/deploy.sh` through the root `.env` file.

For each demo Clerk user, set `public_metadata.pmrf_person_id` and include
`"pmrf_person_id": "{{user.public_metadata.pmrf_person_id}}"` in the session-token customization. The deployed
gateway allowlist is `MOBILE_DEMO_PERSON_IDS` (default: `cgmacros-003,cgmacros-002`).

For Expo Web, set `MOBILE_CORS_ORIGINS` in that file to the comma-separated browser origins that may call the
gateway. The development defaults allow `http://localhost:8081` and `http://localhost:8082`; native clients do not
use browser CORS.
