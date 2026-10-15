import { Text, TextInput, View, type TextInputProps } from 'react-native';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type QuestionnaireFieldProps = TextInputProps & {
  label: string;
  suffix?: string;
};

export function QuestionnaireField({ label, suffix, ...props }: QuestionnaireFieldProps) {
  const { colors, controlStyle, text } = usePatientTheme();
  return (
    <View className="flex-1">
      <Text className="mb-2 text-base text-copy" style={[text.body, { color: colors.mutedForeground }]}>{label}</Text>
      <View className="flex-row items-center rounded-card px-4" style={[controlStyle, { backgroundColor: colors.background }]}>
        <TextInput className="flex-1 py-4 text-lg text-ink" placeholderTextColor={colors.mutedForeground} style={[text.body, { color: colors.foreground }]} {...props} />
        {suffix ? <Text className="text-base text-copy" style={[text.body, { color: colors.mutedForeground }]}>{suffix}</Text> : null}
      </View>
    </View>
  );
}
