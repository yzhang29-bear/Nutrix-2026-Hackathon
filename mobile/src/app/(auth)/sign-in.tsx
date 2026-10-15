import { useSignIn } from '@clerk/expo';
import { router } from 'expo-router';
import { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { AuthHeader, AuthScreen, ErrorMessage, TextButton } from '@/components/auth/auth-ui';
import { messageForClerkError } from '@/auth/clerk-error';
import { colors } from '../../../constants/theme';

type FieldErrors = {
  email?: string;
  password?: string;
};

export default function SignInScreen() {
  const { signIn, fetchStatus } = useSignIn();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isBusy = isSubmitting || fetchStatus === 'fetching';

  async function submit() {
    setSubmitError(undefined);
    const nextErrors: FieldErrors = {};
    if (!email.trim()) nextErrors.email = 'Enter your email address.';
    if (!password) nextErrors.password = 'Enter your password.';
    if (Object.keys(nextErrors).length) {
      setFieldErrors(nextErrors);
      return;
    }

    setIsSubmitting(true);
    try {
      const { error: signInFailure } = await signIn.password({
        emailAddress: email.trim().toLowerCase(),
        password,
      });
      if (signInFailure) {
        setSubmitError(messageForClerkError(signInFailure));
        return;
      }
      if (signIn.status === 'needs_new_password') {
        router.replace({ pathname: '/(auth)/reset-password', params: { email: email.trim().toLowerCase() } });
        return;
      }
      if (signIn.status === 'needs_second_factor' || signIn.status === 'needs_client_trust') {
        router.replace('/(auth)/second-factor');
        return;
      }
      if (signIn.status !== 'complete') {
        setSubmitError('This sign-in needs another verification step. Use your invitation link to finish setup, then try again.');
        return;
      }

      const { error: finalizeFailure } = await signIn.finalize({ navigate: () => router.replace('/') });
      if (finalizeFailure) {
        setSubmitError(messageForClerkError(finalizeFailure));
      }
    } catch (reason) {
      setSubmitError(reason instanceof Error ? reason.message : 'We could not sign you in.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthScreen>
      <AuthHeader title="Secure login" subtitle="Sign in with your Nutrix email and password." />

      <View style={styles.form}>
        <View style={styles.field}>
          <Text style={styles.label}>Email address</Text>
          <TextInput
            accessibilityLabel="Email address"
            autoCapitalize="none"
            autoComplete="email"
            autoCorrect={false}
            keyboardType="email-address"
            onChangeText={(value) => {
              setEmail(value);
              setFieldErrors((current) => ({ ...current, email: undefined }));
            }}
            placeholder="Enter your email"
            placeholderTextColor="rgba(0, 0, 0, 0.45)"
            style={[styles.input, fieldErrors.email && styles.inputError]}
            textContentType="emailAddress"
            value={email}
          />
          {fieldErrors.email ? <Text style={styles.fieldError}>{fieldErrors.email}</Text> : null}
        </View>

        <View style={styles.field}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            accessibilityLabel="Password"
            autoComplete="current-password"
            autoCorrect={false}
            onChangeText={(value) => {
              setPassword(value);
              setFieldErrors((current) => ({ ...current, password: undefined }));
            }}
            onSubmitEditing={submit}
            placeholder="Enter your password"
            placeholderTextColor="rgba(0, 0, 0, 0.45)"
            secureTextEntry
            style={[styles.input, fieldErrors.password && styles.inputError]}
            textContentType="password"
            value={password}
          />
          {fieldErrors.password ? <Text style={styles.fieldError}>{fieldErrors.password}</Text> : null}
        </View>

        <ErrorMessage message={submitError} />
        <Pressable
          accessibilityLabel="Sign in"
          accessibilityRole="button"
          disabled={isBusy}
          onPress={submit}
          style={({ pressed }) => [styles.signInButton, (pressed || isBusy) && styles.buttonMuted]}>
          {isBusy ? <ActivityIndicator color={colors.onPrimary} /> : <Text style={styles.signInButtonText}>Sign in</Text>}
        </Pressable>
      </View>

      <TextButton label="Forgot password?" onPress={() => router.push('/(auth)/forgot-password')} />
      <View style={styles.linkRow}>
        <Text style={styles.linkCopy}>New to Nutrix?</Text>
        <Pressable accessibilityRole="link" onPress={() => router.push('/(auth)/sign-up')}>
          <Text style={styles.link}>Create a patient account</Text>
        </Pressable>
      </View>
    </AuthScreen>
  );
}

const styles = StyleSheet.create({
  form: { gap: 16 },
  field: { gap: 8 },
  label: { color: colors.foreground, fontSize: 16, fontWeight: '700' },
  input: { height: 54, borderWidth: 1, borderColor: colors.border, borderRadius: 14, color: colors.foreground, fontSize: 17, paddingHorizontal: 15, backgroundColor: colors.card },
  inputError: { borderColor: colors.destructive },
  fieldError: { color: colors.destructive, fontSize: 13, fontWeight: '600' },
  signInButton: { minHeight: 56, borderRadius: 14, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.primary, marginTop: 2, elevation: 3 },
  signInButtonText: { color: colors.onPrimary, fontSize: 17, fontWeight: '800' },
  buttonMuted: { opacity: 0.65 },
  linkRow: { flexDirection: 'row', alignItems: 'center', alignSelf: 'center', gap: 5, marginTop: 2, paddingVertical: 10 },
  linkCopy: { color: colors.mutedForeground, fontSize: 15 },
  link: { color: colors.primary, fontSize: 15, fontWeight: '800' },
});
