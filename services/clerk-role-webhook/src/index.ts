import { createClerkClient } from '@clerk/backend';
import { verifyWebhook } from '@clerk/backend/webhooks';

type LambdaWebhookEvent = {
  body?: string | null;
  headers: Record<string, string | undefined>;
  isBase64Encoded?: boolean;
  rawPath?: string;
  requestContext: {
    domainName?: string;
    http: { method: string };
  };
};

const staffRoles = new Set(['admin', 'clinician']);

function requiredEnvironment(name: 'CLERK_SECRET_KEY' | 'CLERK_WEBHOOK_SIGNING_SECRET') {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be configured.`);
  return value;
}

function requestFromLambdaEvent(event: LambdaWebhookEvent) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(event.headers)) {
    if (value) headers.set(name, value);
  }

  const body = event.body
    ? event.isBase64Encoded
      ? Buffer.from(event.body, 'base64')
      : event.body
    : undefined;

  return new Request(
    `https://${event.requestContext.domainName ?? 'localhost'}${event.rawPath ?? '/'}`,
    { body, headers, method: event.requestContext.http.method },
  );
}

/**
 * Public Lambda Function URL handler for Clerk's `user.created` webhook.
 * The Clerk signature is verified before any metadata is written.
 */
export async function handler(event: LambdaWebhookEvent) {
  try {
    const webhook = await verifyWebhook(requestFromLambdaEvent(event), {
      signingSecret: requiredEnvironment('CLERK_WEBHOOK_SIGNING_SECRET'),
    });

    if (webhook.type === 'user.created') {
      const existingRole = String(webhook.data.public_metadata?.role ?? '').toLowerCase();

      // Staff created through a trusted backend may already carry its role. Every
      // other newly created account is a Patient; this is the least-privileged role.
      if (!staffRoles.has(existingRole)) {
        const clerk = createClerkClient({
          secretKey: requiredEnvironment('CLERK_SECRET_KEY'),
        });
        await clerk.users.updateUserMetadata(webhook.data.id, {
          publicMetadata: { role: 'patient' },
        });
      }
    }

    return { body: 'OK', statusCode: 200 };
  } catch (error) {
    console.error('Clerk role webhook failed', error);
    return { body: 'Webhook verification failed', statusCode: 400 };
  }
}
