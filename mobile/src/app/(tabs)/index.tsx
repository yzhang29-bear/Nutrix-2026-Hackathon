import { ActivityIndicator, Pressable, ScrollView, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { PhotoPickerStatusMessage } from '@/components/meals/meal-photo-preview';
import { useMealDraft } from '@/features/meals/meal-draft-context';
import { useMealPhotoPicker } from '@/hooks/use-meal-photo-picker';
import { createGatewayClient, type Readiness } from '@/api/gateway';
import { useAuth } from '@/auth/auth-provider';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export default function TodayScreen() {
  const router = useRouter();
  const { startDraft } = useMealDraft();
  const { getSessionToken } = useAuth();
  const { t } = useLanguage();
  const { cardStyle, colors, primaryButtonStyle, text } = usePatientTheme();
  const gateway = useMemo(() => createGatewayClient({ getToken: getSessionToken }), [getSessionToken]);
  const [readiness, setReadiness] = useState<Readiness>();
  const [readinessError, setReadinessError] = useState<string>();

  const loadReadiness = useCallback(async () => {
    try {
      setReadinessError(undefined);
      setReadiness(await gateway.getReadiness());
    } catch (error) {
      setReadinessError(error instanceof Error ? error.message : 'Unable to load your Nutrix status.');
    }
  }, [gateway]);

  useEffect(() => {
    void loadReadiness();
  }, [loadReadiness]);
  const { choosePhotoSource, status } = useMealPhotoPicker({
    onPhotoReady: (photo) => {
      startDraft(photo);
      router.push('/meal-edit');
    },
  });

  return (
    <SafeAreaView className="app-screen px-5 pt-5" style={{ backgroundColor: colors.background }}>
      <ScrollView contentContainerClassName="pb-32">
        <Text className="font-semibold uppercase tracking-widest text-brand-dark" style={[text.label, { color: colors.onPrimarySoft }]}>
          Nutrix
        </Text>
        <Text className="mt-3 font-bold tracking-tight text-ink" style={[text.pageTitle, { color: colors.foreground }]}>{t('today')}</Text>
        <Text className="app-body-copy mt-3" style={[text.body, { color: colors.mutedForeground }]}>
          {t('todayDescription')}
        </Text>
        <View className="app-card mt-5" style={cardStyle}>
          <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('personalizationStatus')}</Text>
          {readiness ? (
            <Text className="mt-1 text-sm leading-5 text-copy" style={[text.label, { color: colors.mutedForeground }]}>
              {readiness.status === 'ready'
                ? t('modelReady')
                : t('daysCollected', { days: readiness.daysCollected, required: readiness.requiredDays })}
            </Text>
          ) : (
            <Text className="mt-1 text-sm leading-5 text-copy" style={[text.label, { color: colors.mutedForeground }]}>{readinessError ?? t('checkingStatus')}</Text>
          )}
        </View>
        <View className="app-card mt-8" style={cardStyle}>
          <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('mealPhoto')}</Text>
          <Text className="mt-1 text-sm leading-5 text-copy" style={[text.label, { color: colors.mutedForeground }]}>
            {t('addPhotoDescription')}
          </Text>

          <Pressable
            accessibilityLabel={t('addMealPhoto')}
            className="app-button-primary mt-4"
            disabled={status === 'processing'}
            onPress={choosePhotoSource}
            style={primaryButtonStyle}>
            {status === 'processing' ? (
              <ActivityIndicator color={colors.onPrimary} />
            ) : (
              <Text className="font-semibold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>
                {t('addMealPhoto')}
              </Text>
            )}
          </Pressable>
          {status === 'processing' ? <Text className="mt-3 text-center text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('preparingPhoto')}</Text> : null}
          <PhotoPickerStatusMessage status={status} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
