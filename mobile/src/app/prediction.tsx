import { useEffect, useMemo } from 'react';
import { useRouter } from 'expo-router';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { MetricCard } from '@/components/prediction/metric-card';
import { PredictionChart } from '@/components/prediction/prediction-chart';
import { PredictionDrivers } from '@/components/prediction/prediction-drivers';
import { PredictionHeader } from '@/components/prediction/prediction-header';
import { useMealDraft } from '@/features/meals/meal-draft-context';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export default function PredictionScreen() {
  const router = useRouter();
  const { discardDraft, prediction } = useMealDraft();
  const { t } = useLanguage();
  const { cardStyle, colors, primaryButtonStyle, text } = usePatientTheme();

  useEffect(() => {
    if (!prediction) router.replace('/(tabs)');
  }, [prediction, router]);

  const chartData = useMemo(() => prediction && ({
    observed: prediction.observed?.curve.glucose_mgdl ?? prediction.prediction.curve.glucose_mgdl,
    predicted: prediction.prediction.curve.glucose_mgdl,
    lowerBound: prediction.prediction.curve.lower80,
    upperBound: prediction.prediction.curve.upper80,
    minutes: prediction.prediction.curve.minutes,
  }), [prediction]);

  if (!prediction || !chartData) return null;

  const comparisonRows = [30, 60, 90, 120].map((minute) => ({
    minute: `${minute} min`,
    observed: prediction.observed?.glucose_at_min[String(minute)],
    predicted: prediction.prediction.glucose_at_min[String(minute)],
  }));
  const largestContribution = Math.max(...prediction.explanation.contributions.map((item) => Math.abs(item.mgdl)), 1);
  const drivers = prediction.explanation.contributions.map((item) => ({
    label: item.factor,
    value: Math.round((Math.abs(item.mgdl) / largestContribution) * 100),
  }));

  function returnHome() {
    discardDraft();
    router.replace('/(tabs)');
  }

  return (
    <SafeAreaView className="app-screen" edges={['top']} style={{ backgroundColor: colors.background }}>
      <ScrollView contentContainerClassName="px-5 pb-12">
        <PredictionHeader onBack={() => router.back()} title={t('mealPrediction')} />
        <PredictionChart
          data={chartData}
          day={8}
          peakGlucose={prediction.prediction.peak_glucose_mgdl}
          timeToPeak={prediction.prediction.time_to_peak_min}
        />

        <View className="mt-4 flex-row gap-3">
          <MetricCard label={t('peakLabel')} unit="mg/dL" value={String(Math.round(prediction.prediction.peak_glucose_mgdl))} />
          <MetricCard label={t('timeToPeak')} unit="min" value={String(Math.round(prediction.prediction.time_to_peak_min))} />
          <MetricCard label="2h iAUC" unit="mg·min/dL" value={Math.round(prediction.prediction.iauc_0_120_mgdl_min).toLocaleString()} />
        </View>

        <View className="app-card mt-4" style={cardStyle}>
          <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{t('glucoseAfterMeal')}</Text>
          <Text className="mt-1 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('predictedObserved')}</Text>
          <View className="mt-4 flex-row justify-between">
            {comparisonRows.map((row) => (
              <View className="items-center" key={row.minute}>
                <Text className="text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>{row.minute}</Text>
                <Text className="mt-1 text-2xl font-semibold text-prediction" style={[text.reading, { color: colors.medication }]}>{Math.round(row.predicted)}</Text>
                {row.observed !== undefined ? <Text className="mt-1 text-xs font-medium text-glucose" style={[text.caption, { color: colors.success }]}>obs {Math.round(row.observed)}</Text> : null}
              </View>
            ))}
          </View>
        </View>

        <PredictionDrivers drivers={drivers} />

        <View className="mt-4 rounded-card bg-status-low px-5 py-4" style={{ backgroundColor: colors.statusLow }}>
          <Text className="text-base font-semibold text-on-secondary" style={[text.body, { color: colors.onSecondary }]}>{t('confidence')}: {prediction.confidence.level}</Text>
          <Text className="mt-1 text-sm leading-5 text-on-secondary" style={[text.label, { color: colors.onSecondary }]}>
            {prediction.confidence.flags.length ? prediction.confidence.flags.join(' · ') : 'Personalized from your available data.'}
          </Text>
        </View>

        <Text className="mt-5 text-center text-xs leading-5 text-copy" style={[text.caption, { color: colors.mutedForeground }]}>
          This is a demonstration estimate and not medical advice.
        </Text>
        <Pressable accessibilityRole="button" className="app-button-primary mt-5" onPress={returnHome} style={primaryButtonStyle}>
          <Text className="text-lg font-bold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>{t('goHome')}</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}
