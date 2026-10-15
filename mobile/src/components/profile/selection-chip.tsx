import { Pressable, Text } from 'react-native';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type SelectionChipProps = {
  label: string;
  onPress: () => void;
  selected: boolean;
};

export function SelectionChip({ label, onPress, selected }: SelectionChipProps) {
  const { colors, controlStyle, text } = usePatientTheme();
  return (
    <Pressable
      accessibilityRole="checkbox"
      accessibilityState={{ checked: selected }}
      className="min-h-12 justify-center rounded-card border px-4 py-2"
      onPress={onPress}
      style={[controlStyle, { backgroundColor: selected ? colors.secondary : colors.background, borderColor: selected ? colors.primary : 'transparent' }]}>
      <Text className={selected ? 'font-semibold' : ''} style={[text.body, { color: selected ? colors.onSecondary : colors.foreground }]}>{label}</Text>
    </Pressable>
  );
}
