export const roles = ['Admin', 'Clinician', 'Patient'] as const;

export type Role = (typeof roles)[number];

export type DemoPersonId = 'cgmacros-002' | 'cgmacros-003';

export type AuthenticatedUser = {
  email: string;
  name: string;
  personId?: DemoPersonId;
  role: Role;
};

/** Returns the supported CGMacros participant assigned to this Clerk user. */
export function demoPersonIdFromMetadata(metadata: unknown): DemoPersonId | undefined {
  const personId =
    typeof metadata === 'object' && metadata && 'pmrf_person_id' in metadata
      ? String(metadata.pmrf_person_id).toLowerCase()
      : '';

  return personId === 'cgmacros-002' || personId === 'cgmacros-003' ? personId : undefined;
}

/**
 * Staff roles are written to Clerk public metadata by a trusted admin process.
 * Accounts without a staff role are patients: public registration is patient-only.
 */
export function roleFromMetadata(metadata: unknown): Role {
  const role =
    typeof metadata === 'object' && metadata && 'role' in metadata
      ? String(metadata.role).toLowerCase()
      : '';

  if (role === 'admin') return 'Admin';
  if (role === 'clinician') return 'Clinician';
  return 'Patient';
}
