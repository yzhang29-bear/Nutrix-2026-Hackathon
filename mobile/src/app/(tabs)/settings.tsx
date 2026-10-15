import { useState } from 'react';
import { Alert, Pressable, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from "react-native-safe-area-context";

import { useAuth } from '@/auth/auth-provider';
import { LanguageSelector } from '@/components/settings/language-selector';
import { SeniorModeToggle } from '@/components/settings/senior-mode-toggle';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export default function SettingsScreen() {
  const { signOut, user } = useAuth();
  const router = useRouter();
  const { language, setLanguage, t } = useLanguage();
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();
  const [isLanguageMenuOpen, setIsLanguageMenuOpen] = useState(false);

  function requestSignOut() {
    Alert.alert(t('signOutTitle'), t('signOutBody'), [
      { text: t('cancel'), style: 'cancel' },
      { text: t('signOut'), style: 'destructive', onPress: () => void signOut() },
    ]);
  }

  return (
    <SafeAreaView className="app-screen px-5 pt-5" style={{ backgroundColor: colors.background }}>
      <Text className="font-semibold uppercase tracking-widest text-brand-dark" style={[text.label, { color: colors.onPrimarySoft }]}>
        Nutrix
      </Text>
      <Text className="mt-3 font-bold tracking-tight text-ink" style={[text.pageTitle, { color: colors.foreground }]}>
        {t('settings')}
      </Text>
      <Text className="app-body-copy mt-3" style={[text.body, { color: colors.mutedForeground }]}>
        Manage your preferences and health-data connections.
      </Text>
      <View className="app-card mt-8" style={cardStyle}>
        <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>
          {user?.name ?? t('accountFallback')}
        </Text>
        <Text className="mt-1 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{user?.email}</Text>
        <View className="mt-3 self-start rounded-full bg-brand-soft px-3 py-1" style={{ backgroundColor: colors.primarySoft }}>
          <Text className="text-xs font-bold uppercase tracking-wide text-brand-dark" style={[text.caption, { color: colors.onPrimarySoft }]}>{user?.role}</Text>
        </View>
      </View>
      <View className="app-card mt-4" style={cardStyle}>
        <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>
          {t('preferences')}
        </Text>
        <Text className="mt-2 text-sm leading-5 text-copy" style={[text.label, { color: colors.mutedForeground }]}>
          {t('preferencesDescription')}
        </Text>
      </View>
      <Pressable accessibilityRole="button" className="app-card mt-4 flex-row items-center justify-between" onPress={() => router.push('/profile')} style={cardStyle}>
        <View>
          <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{t('healthProfile')}</Text>
          <Text className="mt-1 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('healthProfileDescription')}</Text>
        </View>
        <Text className="text-2xl text-brand-dark" style={{ color: colors.onPrimarySoft }}>›</Text>
      </Pressable>
      <LanguageSelector
        description={t('languageDescription')}
        isOpen={isLanguageMenuOpen}
        onClose={() => setIsLanguageMenuOpen(false)}
        onOpen={() => setIsLanguageMenuOpen(true)}
        onSelect={(nextLanguage) => {
          setLanguage(nextLanguage);
          setIsLanguageMenuOpen(false);
        }}
        selectedLanguage={language}
        title={t('language')}
      />
      <SeniorModeToggle description={t('seniorModeDescription')} title={t('seniorMode')} />
      <Pressable
        accessibilityRole="button"
        onPress={requestSignOut}
        className="mt-6 items-center rounded-2xl border border-red-200 bg-red-50 px-5 py-4"
        style={[{ backgroundColor: colors.statusHigh, borderColor: colors.destructive }, controlStyle]}>
        <Text className="text-base font-bold text-danger" style={[text.button, { color: colors.destructive }]}>{t('signOut')}</Text>
      </Pressable>
    </SafeAreaView>
  );
}
