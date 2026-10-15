import { Text, View } from 'react-native';

type PatientSummaryReportProps = {
  averageGlucose: number;
  highRange: number;
  latestDayAverage: number;
  lowRange: number;
  meals: number;
  timeInRange: number;
  t: (key: string, values?: Record<string, string | number>) => string;
};

function ReportRow({ label, value }: { label: string; value: string }) {
  return (
    <View className="flex-row items-center justify-between border-b border-line py-3 last:border-b-0">
      <Text className="text-base text-copy">{label}</Text>
      <Text className="font-numeric text-base font-bold text-ink">{value}</Text>
    </View>
  );
}

export function PatientSummaryReport({ averageGlucose, highRange, latestDayAverage, lowRange, meals, timeInRange, t }: PatientSummaryReportProps) {
  return (
    <View className="app-card mt-4">
      <Text className="text-xl font-bold text-ink">{t('sevenDaySummary')}</Text>
      <Text className="mt-1 text-sm text-copy">{t('summaryReportDescription')}</Text>
      <View className="mt-3">
        <ReportRow label={t('averageGlucose')} value={`${averageGlucose} mg/dL`} />
        <ReportRow label={t('timeInRange')} value={`${timeInRange}%`} />
        <ReportRow label={t('lowRange')} value={`${lowRange}%`} />
        <ReportRow label={t('highRange')} value={`${highRange}%`} />
        <ReportRow label={t('recordedMeals')} value={String(meals)} />
        <ReportRow label={t('latestDayAverage')} value={`${latestDayAverage} mg/dL`} />
      </View>
    </View>
  );
}
