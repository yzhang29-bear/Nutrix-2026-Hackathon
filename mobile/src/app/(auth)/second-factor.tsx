import { router } from "expo-router";
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

function factorDescription(factor: {
  strategy: string;
  safeIdentifier?: string;
}) {
  return `${factor.strategy.replace("_", " ")}${factor.safeIdentifier ? ` (${factor.safeIdentifier})` : ""}`;
}

export default function SecondFactorScreen() {
  const { signIn, fetchStatus } = useSignIn();
  const factors = (signIn.supportedSecondFactors ?? []).filter((factor) =>
    ["email_code", "phone_code", "totp", "backup_code"].includes(
      factor.strategy,
    ),
  );
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const selectedFactor =
    factors.find((factor) => factor.strategy === selectedStrategy) ??
    factors[0];
  const [code, setCode] = useState("");
  const [codeSent, setCodeSent] = useState(false);
  const [error, setError] = useState<string>();
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function sendCode() {
    if (!selectedFactor) return;
    setError(undefined);
    try {
      const result =
        selectedFactor.strategy === "email_code"
          ? await signIn.mfa.sendEmailCode()
          : selectedFactor.strategy === "phone_code"
            ? await signIn.mfa.sendPhoneCode()
            : undefined;
      if (result?.error) {
        setError(messageForClerkError(result.error));
        return;
      }
      setCodeSent(true);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "We could not send a verification code.",
      );
    }
  }

  async function verify() {
    if (!selectedFactor || !code.trim()) {
      setError("Enter your verification code.");
      return;
    }
    setError(undefined);
    setIsSubmitting(true);
    try {
      const result =
        selectedFactor.strategy === "email_code"
          ? await signIn.mfa.verifyEmailCode({ code: code.trim() })
          : selectedFactor.strategy === "phone_code"
            ? await signIn.mfa.verifyPhoneCode({ code: code.trim() })
            : selectedFactor.strategy === "totp"
              ? await signIn.mfa.verifyTOTP({ code: code.trim() })
              : await signIn.mfa.verifyBackupCode({ code: code.trim() });
      if (result.error) {
        setError(messageForClerkError(result.error));
        return;
      }
      if (signIn.status === "complete") {
        const { error: finalizeFailure } = await signIn.finalize({
          navigate: () => router.replace("/"),
        });
        if (finalizeFailure) setError(messageForClerkError(finalizeFailure));
      } else {
        setError("Verification is not complete. Check the code and try again.");
      }
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "We could not verify your sign-in.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  const usesCodeDelivery =
    selectedFactor?.strategy === "email_code" ||
    selectedFactor?.strategy === "phone_code";

  return (
    <AuthScreen>
      <AuthHeader
        title="Verify your sign-in"
        subtitle={signIn.status === 'needs_client_trust' ? 'Verify this new device to continue.' : 'Complete the additional verification step to continue.'}
      />
      <ErrorMessage message={error} />
      {!selectedFactor ? (
        <>
          <ErrorMessage message="No supported second-factor method is available. Return to sign-in and try again." />
          <TextButton
            label="Return to sign in"
            onPress={() => router.replace("/(auth)/sign-in")}
          />
        </>
      ) : (
        <>
          {factors.map((factor) => (
            <TextButton
              key={`${factor.strategy}-${"safeIdentifier" in factor ? factor.safeIdentifier : ""}`}
              label={factorDescription(factor)}
              onPress={() => {
                setSelectedStrategy(factor.strategy);
                setCode("");
                setCodeSent(false);
                setError(undefined);
              }}
            />
          ))}
          {usesCodeDelivery && !codeSent ? (
            <PrimaryButton
              label={`Send code to ${factorDescription(selectedFactor)}`}
              onPress={sendCode}
              disabled={isSubmitting || fetchStatus === "fetching"}
            />
          ) : (
            <>
              <Field
                label={factorDescription(selectedFactor)}
                value={code}
                onChangeText={setCode}
                keyboardType="number-pad"
                autoComplete="one-time-code"
                placeholder="Enter verification code"
                returnKeyType="done"
                onSubmitEditing={verify}
              />
              <PrimaryButton
                label="Verify"
                onPress={verify}
                disabled={isSubmitting || fetchStatus === "fetching"}
              />
            </>
          )}
          <TextButton
            label="Return to sign in"
            onPress={() => router.replace("/(auth)/sign-in")}
          />
        </>
      )}
    </AuthScreen>
  );
}
