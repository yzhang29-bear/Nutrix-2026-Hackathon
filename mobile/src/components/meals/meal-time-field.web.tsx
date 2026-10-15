import { Pressable, Text, View } from 'react-native';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type MealTimeFieldProps = {
  onChange: (time: Date) => void;
  value: Date;
};

function formatTime(time: Date) {
  return time.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

export function MealTimeField({ onChange, value }: MealTimeFieldProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();
  return (
    <View className="app-card flex-1" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('mealTime')}</Text>
      <Pressable
        accessibilityLabel={t('mealTime')}
        accessibilityRole="button"
        className="mt-3 rounded-card px-4 py-4"
        onPress={() => onChange(new Date())}
        style={[controlStyle, { backgroundColor: colors.background }]}>
        <Text className="font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{formatTime(value)}</Text>
        <Text className="mt-1 text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>{t('currentTimePreview')}</Text>
      </Pressable>
    </View>
  );
}
