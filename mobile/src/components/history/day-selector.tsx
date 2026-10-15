import { Pressable, Text, View } from 'react-native';

import type { HistoryDayId } from '@/data/history';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type DaySelectorProps = {
  days: readonly HistoryDayId[];
  onSelect: (day: HistoryDayId) => void;
  selectedDay: HistoryDayId;
};

export function DaySelector({ days, onSelect, selectedDay }: DaySelectorProps) {
  const { colors, isSeniorMode, text } = usePatientTheme();
  return (
    <View className="flex-row justify-between gap-2">
      {days.map((day) => {
        const selected = day === selectedDay;

        return (
          <Pressable
            key={day}
            accessibilityLabel={`Select ${day}`}
            accessibilityRole="button"
            accessibilityState={{ selected }}
            className="flex-1 items-center justify-center rounded-2xl"
            onPress={() => onSelect(day)}
            style={({ pressed }) => ({
              backgroundColor: selected ? colors.primary : colors.card,
              minHeight: isSeniorMode ? 56 : 48,
              opacity: pressed ? 0.78 : 1,
              shadowColor: colors.foreground,
              shadowOffset: { width: 0, height: 4 },
              shadowOpacity: selected ? 0.16 : 0.06,
              shadowRadius: 8,
              elevation: 2,
            })}>
            <Text className="font-semibold" style={[text.label, { color: selected ? colors.onPrimary : colors.foreground }]}>{day}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}
