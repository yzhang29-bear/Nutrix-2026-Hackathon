import { useState } from 'react';
import { ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DailyAverageChart } from '@/components/history/daily-average-chart';
import { DaySelector } from '@/components/history/day-selector';
import { HistoryTimelineChart } from '@/components/history/history-timeline-chart';
import { LoggedMealsList } from '@/components/history/logged-meals-list';
import { getDemoHistory } from '@/data/demo-history';
import type { HistoryDayId } from '@/data/history';
import { useAuth } from '@/auth/auth-provider';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export default function HistoryScreen() {
  const { user } = useAuth();
  const { t } = useLanguage();
  const { colors, text } = usePatientTheme();
  const history = getDemoHistory(user?.personId);
  const [selectedDayId, setSelectedDayId] = useState<HistoryDayId>('D7');
  const selectedDay = history.days.find((day) => day.id === selectedDayId) ?? history.latestDay;

  return (
    <SafeAreaView className="app-screen" edges={['top']} style={{ backgroundColor: colors.background }}>
      <ScrollView contentContainerClassName="app-scroll-content" showsVerticalScrollIndicator={false}>
        <Text className="app-page-title" style={[text.pageTitle, { color: colors.foreground }]}>{t('history')}</Text>
        <View className="mt-2">
          <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>
            {history.participantLabel} · 7-day CGMacros baseline · {history.sourceWindow}
          </Text>
        </View>
        <View className="mt-5">
          <DaySelector days={history.days.map((day) => day.id)} onSelect={setSelectedDayId} selectedDay={selectedDayId} />
        </View>
        <HistoryTimelineChart day={selectedDay} />
        <DailyAverageChart days={history.days} selectedDay={selectedDayId} />
        {selectedDay.meals.length > 0 ? <LoggedMealsList day={selectedDay} /> : null}
        <Text className="mt-5 text-center text-xs leading-5 text-copy" style={[text.caption, { color: colors.mutedForeground }]}>
          {t('historyFootnote')}
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}
