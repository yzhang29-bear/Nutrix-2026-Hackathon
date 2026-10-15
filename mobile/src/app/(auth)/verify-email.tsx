import { router, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { useSignUp } from '@clerk/expo';

import { AuthHeader, AuthScreen, ErrorMessage, Field, PrimaryButton, TextButton } from '@/components/auth/auth-ui';
import { messageForClerkError } from '@/auth/clerk-error';
import { useHealthQuestionnaire } from '@/features/profile/health-questionnaire-context';

export default function VerifyEmailScreen() {
  const { email: rawEmail } = useLocalSearchParams<{ email?: string }>();
  const email = Array.isArray(rawEmail) ? rawEmail[0] : rawEmail ?? '';
  const { signUp, fetchStatus } = useSignUp();
  const { beginQuestionnaire, completeQuestionnaire } = useHealthQuestionnaire();
  const [code, setCode] = useState('');
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function verify() {
    setError(undefined);
    if (!email || !code.trim()) {
      setError('Enter the verification code sent to your email.');
      return;
    }
    setIsSubmitting(true);
    try {
      const { error: verificationFailure } = await signUp.verifications.verifyEmailCode({ code: code.trim() });
      if (verificationFailure) {
        setError(messageForClerkError(verificationFailure));
        return;
      }
      if (signUp.status !== 'complete') {
        setError('Your email is verified, but your account setup is not complete. Please sign in.');
        router.replace('/(auth)/sign-in');
        return;
      }
      beginQuestionnaire();
      const { error: finalizeFailure } = await signUp.finalize({ navigate: () => router.replace('/') });
      if (finalizeFailure) {
        completeQuestionnaire();
        setError(messageForClerkError(finalizeFailure));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'We could not verify that code.');
    } finally {
      setIsSubmitting(false);
    }
  }

  async function resend() {
    setError(undefined);
    setNotice(undefined);
    try {
      const { error: resendFailure } = await signUp.verifications.sendEmailCode();
      if (resendFailure) {
        setError(messageForClerkError(resendFailure));
        return;
      }
      setNotice('A new verification code has been sent.');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'We could not send a new code.');
    }
  }

  return (
    <AuthScreen>
      <AuthHeader title="Verify your email" subtitle={email ? `Enter the code sent to ${email}.` : 'Enter the code sent to your email.'} />
      <ErrorMessage message={error ?? notice} />
      <Field label="Verification code" value={code} onChangeText={setCode} keyboardType="number-pad" autoComplete="one-time-code" textContentType="oneTimeCode" placeholder="123456" returnKeyType="done" onSubmitEditing={verify} />
      <PrimaryButton label="Verify and continue" onPress={verify} disabled={isSubmitting || fetchStatus === 'fetching'} />
      <TextButton label="Send a new code" onPress={resend} />
      <TextButton label="Back to sign in" onPress={() => router.replace('/(auth)/sign-in')} />
    </AuthScreen>
  );
}
