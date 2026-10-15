import { Text, View } from 'react-native';

import type { HistoryDay } from '@/data/history';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type LoggedMealsListProps = {
  day: HistoryDay;
};

export function LoggedMealsList({ day }: LoggedMealsListProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, text } = usePatientTheme();
  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('loggedMeals', { day: day.id.slice(1) })}</Text>
      <View className="mt-3 gap-4">
        {day.meals.map((meal) => (
          <View className="flex-row items-center" key={`${meal.name}-${meal.time}`}>
            <View className="mr-3 h-3 w-3 rounded-full" style={{ backgroundColor: colors.meal }} />
            <View className="flex-1">
              <Text className="font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{meal.name}</Text>
              <Text className="mt-0.5 text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>
                {meal.summary} · {meal.time} · {meal.carbs} {t('carbs')}
              </Text>
            </View>
            <View className="ml-3 items-end">
              <Text className="font-medium tracking-tight text-ink" style={[text.reading, { color: colors.foreground }]}>{meal.peak}</Text>
              <Text className="text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>{t('peak')}</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}
