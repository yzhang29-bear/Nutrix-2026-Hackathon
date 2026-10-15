import { router, useLocalSearchParams } from "expo-router";
import { useState } from "react";
import { useSignIn } from "@clerk/expo";

import {
  AuthHeader,
  AuthScreen,
  ErrorMessage,
  Field,
  PrimaryButton,
  TextButton,
} from "@/components/auth/auth-ui";
import { messageForClerkError } from "@/auth/clerk-error";
import {
  meetsPasswordRequirement,
  passwordRequirement,
} from "@/auth/password-policy";

export default function ResetPasswordScreen() {
  const { email: rawEmail } = useLocalSearchParams<{ email?: string }>();
  const email = Array.isArray(rawEmail) ? rawEmail[0] : (rawEmail ?? "");
  const { signIn, fetchStatus } = useSignIn();
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const needsNewPassword = signIn.status === 'needs_new_password';

  async function submit() {
    setError(undefined);
    if ((!needsNewPassword && !code.trim()) || !password || !confirmPassword) {
      setError(needsNewPassword ? 'Enter and confirm your new password.' : 'Enter your code and new password.');
      return;
    }
    if (!meetsPasswordRequirement(password)) {
      setError(passwordRequirement);
      return;
    }
    if (password !== confirmPassword) {
      setError("Your passwords do not match.");
      return;
    }
    setIsSubmitting(true);
    try {
      if (signIn.status !== "needs_new_password") {
        const { error: verificationFailure } =
          await signIn.resetPasswordEmailCode.verifyCode({ code: code.trim() });
        if (verificationFailure) {
          setError(messageForClerkError(verificationFailure));
          return;
        }
      }
      const { error: resetFailure } =
        await signIn.resetPasswordEmailCode.submitPassword({
          password,
          signOutOfOtherSessions: true,
        });
      if (resetFailure) {
        setError(messageForClerkError(resetFailure));
        return;
      }
      if (signIn.status === "complete") {
        const { error: finalizeFailure } = await signIn.finalize({
          navigate: () => router.replace("/"),
        });
        if (finalizeFailure) setError(messageForClerkError(finalizeFailure));
      } else if (signIn.status === "needs_second_factor") {
        router.replace("/(auth)/second-factor");
      } else {
        setError(
          `Your password was reset, but sign-in is not complete (${signIn.status ?? "unknown status"}). Please sign in again.`,
        );
      }
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "We could not reset your password.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function resend() {
    setError(undefined);
    setNotice(undefined);
    try {
      const { error: sendCodeFailure } =
        await signIn.resetPasswordEmailCode.sendCode();
      if (sendCodeFailure) {
        setError(messageForClerkError(sendCodeFailure));
        return;
      }
      setNotice("A new reset code has been sent.");
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "We could not send another code.",
      );
    }
  }

  return (
    <AuthScreen>
      <AuthHeader
        title="Choose a new password"
        subtitle={needsNewPassword ? 'Set a password to finish your account setup.' : email ? `Use the code sent to ${email}.` : 'Use the code sent to your email.'}
      />
      <ErrorMessage message={error ?? notice} />
      {!needsNewPassword ? (
        <Field
          label="Reset code"
          value={code}
          onChangeText={setCode}
          keyboardType="number-pad"
          autoComplete="one-time-code"
          placeholder="123456"
          returnKeyType="next"
        />
      ) : null}
      <Field
        label="New password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoComplete="new-password"
        textContentType="newPassword"
        placeholder="Create a secure password"
        returnKeyType="next"
      />
      <Field
        label="Confirm new password"
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        secureTextEntry
        autoComplete="new-password"
        textContentType="newPassword"
        placeholder="Re-enter your password"
        returnKeyType="done"
        onSubmitEditing={submit}
      />
      <PrimaryButton
        label="Reset password"
        onPress={submit}
        disabled={isSubmitting || fetchStatus === "fetching"}
      />
      {!needsNewPassword ? <TextButton label="Send a new reset code" onPress={resend} /> : null}
    </AuthScreen>
  );
}
