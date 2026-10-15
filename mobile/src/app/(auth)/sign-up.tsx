import { router } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useSignUp } from '@clerk/expo';

import { AuthHeader, AuthScreen, ErrorMessage, Field, PrimaryButton, TextButton } from '@/components/auth/auth-ui';
import { messageForClerkError } from '@/auth/clerk-error';
import { meetsPasswordRequirement, passwordRequirement } from '@/auth/password-policy';
import { colors } from '../../../constants/theme';

export default function SignUpScreen() {
  const { signUp, fetchStatus } = useSignUp();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submit() {
    setError(undefined);
    if (!name.trim() || !email.trim() || !password || !confirmPassword) {
      setError('Complete every field to create your patient account.');
      return;
    }
    if (!meetsPasswordRequirement(password)) {
      setError(passwordRequirement);
      return;
    }
    if (password !== confirmPassword) {
      setError('Your passwords do not match.');
      return;
    }
    setIsSubmitting(true);
    try {
      const { error: signUpFailure } = await signUp.password({
        firstName: name.trim(),
        emailAddress: email.trim().toLowerCase(),
        password,
      });
      if (signUpFailure) {
        setError(messageForClerkError(signUpFailure));
        return;
      }
      const { error: sendCodeFailure } = await signUp.verifications.sendEmailCode();
      if (sendCodeFailure) {
        setError(messageForClerkError(sendCodeFailure));
        return;
      }
      router.replace({ pathname: '/(auth)/verify-email', params: { email } });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'We could not create your account.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthScreen>
      <AuthHeader title="Create your account" subtitle="Patient accounts are ready for self-service registration." />
      <ErrorMessage message={error} />
      <Field label="Display name" value={name} onChangeText={setName} autoComplete="name" textContentType="name" placeholder="How should we address you?" returnKeyType="next" />
      <Field label="Email" value={email} onChangeText={setEmail} keyboardType="email-address" autoComplete="email" textContentType="emailAddress" placeholder="you@example.com" returnKeyType="next" />
      <Field label="Password" value={password} onChangeText={setPassword} secureTextEntry autoComplete="new-password" textContentType="newPassword" placeholder="Create a secure password" returnKeyType="next" />
      <Text accessibilityRole="text" style={styles.passwordHint}>
        Patient passwords must be at least 8 characters.
      </Text>
      <Field label="Confirm password" value={confirmPassword} onChangeText={setConfirmPassword} secureTextEntry autoComplete="new-password" textContentType="newPassword" placeholder="Re-enter your password" returnKeyType="done" onSubmitEditing={submit} />
      <PrimaryButton label="Create patient account" onPress={submit} disabled={isSubmitting || fetchStatus === 'fetching'} />
      <TextButton label="Already have an account? Log in" onPress={() => router.back()} />
      <View nativeID="clerk-captcha" />
    </AuthScreen>
  );
}

const styles = StyleSheet.create({
  passwordHint: {
    color: colors.mutedForeground,
    fontSize: 14,
    lineHeight: 20,
    marginTop: -12,
    marginBottom: 18,
  },
});
