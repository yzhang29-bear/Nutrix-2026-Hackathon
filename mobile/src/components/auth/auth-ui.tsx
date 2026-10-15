import type { PropsWithChildren } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type TextInputProps,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors } from '../../../constants/theme';

export function AuthScreen({ children }: PropsWithChildren) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}>
          <View style={styles.card}>{children}</View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

export function AuthHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <View style={styles.header}>
      <View style={styles.mark} accessibilityElementsHidden>
        <Text style={styles.markText}>N</Text>
      </View>
      <Text style={styles.brand}>NUTRIX</Text>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.subtitle}>{subtitle}</Text>
    </View>
  );
}

export function Field({ label, ...props }: TextInputProps & { label: string }) {
  return (
    <View style={styles.fieldGroup}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.mutedForeground}
        style={styles.input}
        autoCapitalize="none"
        autoCorrect={false}
        {...props}
      />
    </View>
  );
}

export function PrimaryButton({
  label,
  onPress,
  disabled,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [styles.primaryButton, (disabled || pressed) && styles.buttonMuted]}>
      {disabled ? <ActivityIndicator color={colors.onPrimary} /> : <Text style={styles.primaryButtonText}>{label}</Text>}
    </Pressable>
  );
}

export function TextButton({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.textButton}>
      <Text style={styles.textButtonLabel}>{label}</Text>
    </Pressable>
  );
}

export function SecondaryButton({
  label,
  onPress,
}: {
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.secondaryButton, pressed && styles.buttonMuted]}>
      <Text style={styles.secondaryButtonText}>{label}</Text>
    </Pressable>
  );
}

export function ErrorMessage({ message }: { message?: string }) {
  if (!message) {
    return null;
  }
  return (
    <View accessibilityLiveRegion="polite" style={styles.error}>
      <Text style={styles.errorText}>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  scrollContent: { flexGrow: 1, justifyContent: 'center', padding: 20 },
  card: {
    width: '100%',
    maxWidth: 500,
    alignSelf: 'center',
    borderRadius: 28,
    backgroundColor: colors.card,
    paddingHorizontal: 24,
    paddingVertical: 30,
    shadowColor: colors.foreground,
    shadowOpacity: 0.14,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 12 },
    elevation: 6,
  },
  header: { alignItems: 'center', marginBottom: 28 },
  mark: { width: 70, height: 70, borderRadius: 35, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary, marginBottom: 13 },
  markText: { color: colors.onPrimary, fontSize: 32, fontWeight: '800' },
  brand: { color: colors.primary, fontWeight: '800', fontSize: 12, letterSpacing: 2.4, marginBottom: 8 },
  title: { color: colors.foreground, fontSize: 30, fontWeight: '800', textAlign: 'center' },
  subtitle: { color: colors.mutedForeground, fontSize: 16, lineHeight: 23, textAlign: 'center', marginTop: 8 },
  roleTabs: { flexDirection: 'row', borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border, marginBottom: 26 },
  roleTab: { flex: 1, paddingVertical: 13, alignItems: 'center', borderBottomWidth: 3, borderBottomColor: 'transparent' },
  roleTabActive: { borderBottomColor: colors.primary },
  roleTabText: { color: colors.mutedForeground, fontSize: 16, fontWeight: '700' },
  roleTabTextActive: { color: colors.primary },
  fieldGroup: { marginBottom: 17 },
  label: { color: colors.foreground, fontSize: 16, fontWeight: '700', marginBottom: 8 },
  input: { height: 52, borderWidth: 1, borderColor: colors.border, borderRadius: 14, color: colors.foreground, fontSize: 17, paddingHorizontal: 15, backgroundColor: colors.card },
  primaryButton: { minHeight: 52, borderRadius: 14, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.primary, marginTop: 4 },
  primaryButtonText: { color: colors.onPrimary, fontSize: 17, fontWeight: '800' },
  secondaryButton: { minHeight: 52, borderRadius: 14, justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: colors.primary, backgroundColor: colors.primarySoft, marginTop: 10 },
  secondaryButtonText: { color: colors.primary, fontSize: 17, fontWeight: '800' },
  buttonMuted: { opacity: 0.68 },
  textButton: { alignSelf: 'center', paddingVertical: 13, paddingHorizontal: 8 },
  textButtonLabel: { color: colors.primary, fontSize: 15, fontWeight: '700', textAlign: 'center' },
  error: { borderRadius: 12, backgroundColor: '#fef2f2', borderWidth: 1, borderColor: '#fecaca', padding: 12, marginBottom: 17 },
  errorText: { color: '#b91c1c', fontSize: 14, lineHeight: 20 },
  divider: { height: 1, backgroundColor: colors.border, marginVertical: 4 },
});

export function RoleTab({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="tab"
      accessibilityState={{ selected: active }}
      onPress={onPress}
      style={[styles.roleTab, active && styles.roleTabActive]}>
      <Text style={[styles.roleTabText, active && styles.roleTabTextActive]}>{label}</Text>
    </Pressable>
  );
}
