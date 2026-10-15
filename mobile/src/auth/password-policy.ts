export const MIN_PASSWORD_LENGTH = 8;

export const passwordRequirement = `Use at least ${MIN_PASSWORD_LENGTH} characters.`;

export function meetsPasswordRequirement(password: string) {
  return password.length >= MIN_PASSWORD_LENGTH;
}
