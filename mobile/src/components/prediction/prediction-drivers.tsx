import { Text, View } from 'react-native';

import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export type PredictionDriver = {
  label: string;
  value: number;
};

type PredictionDriversProps = {
  drivers: readonly PredictionDriver[];
};

export function PredictionDrivers({ drivers }: PredictionDriversProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, text } = usePatientTheme();
  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('topDrivers')}</Text>
      <View className="mt-4 gap-2">
        {drivers.map((driver, index) => (
          <View className="flex-row items-center" key={driver.label}>
            <Text className="w-44 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{driver.label}</Text>
            <View className="h-2 flex-1 overflow-hidden rounded-full bg-line">
              <View
                className="h-full rounded-full"
                style={{ backgroundColor: index === 0 ? colors.medication : colors.accent, width: `${driver.value}%` }}
              />
            </View>
            <Text className="ml-3 w-9 text-right text-sm font-medium text-ink" style={[text.label, { color: colors.foreground }]}>{driver.value}%</Text>
          </View>
        ))}
      </View>
    </View>
  );
}
