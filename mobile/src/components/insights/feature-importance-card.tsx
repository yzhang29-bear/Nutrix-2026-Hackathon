import { Text, View } from 'react-native';

import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type Feature = {
  label: string;
  value: number;
};

type FeatureImportanceCardProps = {
  features: readonly Feature[];
};

export function FeatureImportanceCard({ features }: FeatureImportanceCardProps) {
  const { cardStyle, colors, text } = usePatientTheme();
  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>Feature importance · this person</Text>
      <View className="mt-4 gap-2">
        {features.map((feature, index) => (
          <View className="flex-row items-center" key={feature.label}>
            <Text className="w-44 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{feature.label}</Text>
            <View className="h-2 flex-1 overflow-hidden rounded-full bg-line">
              <View
                className="h-full rounded-full"
                style={{ backgroundColor: index === 0 ? colors.medication : colors.accent, width: `${feature.value}%` }}
              />
            </View>
            <Text className="ml-3 w-9 text-right text-sm font-medium text-ink" style={[text.label, { color: colors.foreground }]}>{feature.value}%</Text>
          </View>
        ))}
      </View>
    </View>
  );
}
