import { Pressable, ScrollView, Text, View } from 'react-native';
import { Redirect, useLocalSearchParams, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PatientDemographicsCard } from '@/components/clinician/patient-demographics-card';
import { PatientSummaryReport } from '@/components/clinician/patient-summary-report';
import { getClinicianPatient } from '@/data/clinician-patients';
import { getDemoHistory } from '@/data/demo-history';
import { useLanguage } from '@/features/localization/language-context';

export default function ClinicianPatientProfileScreen() {
  const { patientId } = useLocalSearchParams<{ patientId: string }>();
  const patient = getClinicianPatient(patientId ?? '');
  const router = useRouter();
  const { t } = useLanguage();

  if (!patient) return <Redirect href="/(clinician-tabs)" />;

  const history = getDemoHistory(patient.id);
  const averageGlucose = Math.round(history.days.reduce((total, day) => total + day.averageGlucose, 0) / history.days.length);

  return (
    <SafeAreaView className="app-screen" edges={['top']}>
      <ScrollView contentContainerClassName="px-5 pb-12" showsVerticalScrollIndicator={false}>
        <View className="flex-row items-center py-5">
          <Pressable
            accessibilityLabel={t('back')}
            accessibilityRole="button"
            className="mr-3 h-10 w-10 items-center justify-center rounded-full active:bg-line"
            onPress={() => router.back()}>
            <Text className="-mt-1 text-4xl font-light leading-none text-ink">‹</Text>
          </Pressable>
          <Text className="text-3xl font-bold tracking-tight text-ink">{t('patientProfile')}</Text>
        </View>

        <PatientDemographicsCard patient={patient} t={t} />
        <PatientSummaryReport
          averageGlucose={averageGlucose}
          highRange={history.timeInRange.high}
          latestDayAverage={Math.round(history.latestDay.averageGlucose)}
          lowRange={history.timeInRange.low}
          meals={history.trackedMeals}
          t={t}
          timeInRange={history.timeInRange.inRange}
        />
      </ScrollView>
    </SafeAreaView>
  );
}
