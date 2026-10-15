import { Text, View } from 'react-native';

import type { HistoryDay, HistoryDayId } from '@/data/history';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type DailyAverageChartProps = {
  days: readonly HistoryDay[];
  selectedDay: HistoryDayId;
};

export function DailyAverageChart({ days, selectedDay }: DailyAverageChartProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, text } = usePatientTheme();
  const values = days.map((day) => day.averageGlucose);
  const minimum = Math.floor((Math.min(...values) - 10) / 10) * 10;
  const maximum = Math.ceil((Math.max(...values) + 10) / 10) * 10;

  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('dailyAverage')}</Text>
      <View className="mt-4 flex-row items-end border-b border-line px-2 pt-2">
        {days.map((day) => {
          const selected = day.id === selectedDay;
          const height = 28 + ((day.averageGlucose - minimum) / Math.max(1, maximum - minimum)) * 50;

          return (
            <View className="flex-1 items-center" key={day.id}>
              <Text className="mb-1 font-medium text-ink" style={[text.caption, { color: colors.foreground }]}>{day.averageGlucose}</Text>
              <View
                className="w-9 rounded-t-lg"
                style={{ backgroundColor: selected ? colors.primary : colors.secondary, height }}
              />
            </View>
          );
        })}
      </View>
      <View className="mt-2 flex-row px-2">
        {days.map((day) => (
          <View className="flex-1 items-center" key={`label-${day.id}`}>
            <Text className={day.id === selectedDay ? 'font-semibold' : ''} style={[text.caption, { color: day.id === selectedDay ? colors.primary : colors.mutedForeground }]}>
              {day.id}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
}
