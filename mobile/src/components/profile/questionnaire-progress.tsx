import { Text, View } from 'react-native';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export function QuestionnaireProgress({ step }: { step: 1 | 2 }) {
  const { colors, text } = usePatientTheme();
  return (
    <>
      <View className="mt-7 h-1.5 overflow-hidden rounded-full bg-line" style={{ backgroundColor: colors.muted }}>
        <View className="h-full rounded-full bg-brand-hover" style={{ backgroundColor: colors.primary, width: step === 1 ? '50%' : '100%' }} />
      </View>
      <Text className="mt-4 font-bold uppercase tracking-widest text-brand-dark" style={[text.label, { color: colors.onPrimarySoft }]}>Step {step} of 2</Text>
    </>
  );
}
