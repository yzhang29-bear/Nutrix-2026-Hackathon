import { Text, View } from 'react-native';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type MetricCardProps = {
  label: string;
  unit: string;
  value: string;
};

export function MetricCard({ label, value, unit }: MetricCardProps) {
  const { cardStyle, colors, text } = usePatientTheme();
  return (
    <View className="flex-1 rounded-card bg-surface p-4 shadow-card" style={cardStyle}>
      <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{label}</Text>
      <Text className="mt-1 font-semibold tracking-tight text-ink" style={[text.reading, { color: colors.foreground }]}>{value}</Text>
      <Text className="mt-1 text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>{unit}</Text>
    </View>
  );
}
