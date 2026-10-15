type ClerkErrorLike = {
  errors?: { code?: string; longMessage?: string; message?: string }[];
  message?: string;
};

export function messageForClerkError(error: unknown) {
  const first = (error as ClerkErrorLike | null)?.errors?.[0];
  const messages: Record<string, string> = {
    form_identifier_not_found: 'We could not find an account with that email.',
    form_password_incorrect: 'Your email or password is incorrect.',
    form_identifier_exists: 'An account already exists with this email.',
    form_code_incorrect: 'That code is invalid. Please try again.',
    verification_expired: 'That code has expired. Request a new one and try again.',
    form_password_pwned: 'Choose a stronger password that has not appeared in a data breach.',
    form_password_length_too_short: 'Use a password with at least 8 characters.',
  };

  return (first?.code && messages[first.code]) ?? first?.longMessage ?? first?.message ?? 'We could not complete that request. Please try again.';
}
