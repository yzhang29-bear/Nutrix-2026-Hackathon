import { Switch, Text, View } from 'react-native';

import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type SeniorModeToggleProps = {
  description: string;
  title: string;
};

export function SeniorModeToggle({ description, title }: SeniorModeToggleProps) {
  const { cardStyle, colors, isSeniorMode, setSeniorMode, text } = usePatientTheme();

  return (
    <View className="app-card mt-4 flex-row items-center justify-between" style={cardStyle}>
      <View className="mr-4 flex-1">
        <Text className="font-semibold text-ink" style={text.body}>{title}</Text>
        <Text className="mt-1 text-copy" style={text.label}>{description}</Text>
      </View>
      <Switch
        accessibilityLabel={title}
        accessibilityRole="switch"
        onValueChange={setSeniorMode}
        thumbColor={isSeniorMode ? colors.card : colors.card}
        trackColor={{ false: colors.muted, true: colors.primary }}
        value={isSeniorMode}
      />
    </View>
  );
}
