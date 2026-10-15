import { Text, View } from 'react-native';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type ModelConfidenceCardProps = {
  daysTracked: number;
  loggedMeals: number;
  responseWindows: number;
};

export function ModelConfidenceCard({ daysTracked, loggedMeals, responseWindows }: ModelConfidenceCardProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, text } = usePatientTheme();
  return (
    <View className="app-card mt-4" style={cardStyle}>
      <View className="flex-row items-center justify-between">
        <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('recordedCoverage')}</Text>
      </View>
      <Text className="mt-4 text-base leading-6 text-copy" style={[text.body, { color: colors.mutedForeground }]}>
        {t('coverageText', { days: daysTracked, meals: loggedMeals })}
      </Text>
      <Text className="mt-1 text-base leading-6 text-copy" style={[text.body, { color: colors.mutedForeground }]}>
        {t('responseText', { count: responseWindows })}
      </Text>
    </View>
  );
}
