import { useMemo, useState } from 'react';
import { Pressable, Text, TextInput, View } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';

import { clinicianPatients } from '@/data/clinician-patients';
import { useLanguage } from '@/features/localization/language-context';

export default function ClinicianDashboardScreen() {
  const { t } = useLanguage();
  const router = useRouter();
  const [query, setQuery] = useState('');
  const patients = useMemo(
    () => clinicianPatients.filter((patient) => patient.id.includes(query.trim().toLowerCase())),
    [query],
  );

  return (
    <SafeAreaView className="app-screen px-5 pt-5">
      <View className="flex-row items-end justify-between">
        <Text className="text-4xl font-bold tracking-tight text-ink">{t('patients')}</Text>
        <Text className="mb-1 text-lg text-copy">{t('patientCount', { count: clinicianPatients.length })}</Text>
      </View>

      <View className="mt-5 flex-row items-center rounded-card border border-line bg-surface px-4 py-1 shadow-glass">
        <Text className="mr-3 text-2xl text-copy">⌕</Text>
        <TextInput
          accessibilityLabel={t('searchPatients')}
          autoCapitalize="none"
          autoCorrect={false}
          className="flex-1 py-4 text-lg text-ink"
          onChangeText={setQuery}
          placeholder={t('searchPatients')}
          placeholderTextColor="#9aa8a6"
          value={query}
        />
      </View>

      <View className="mt-5 gap-4">
        {patients.map((patient) => (
          <Pressable
            accessibilityHint={t('openPatientProfile')}
            accessibilityLabel={`${patient.name}, ${t('openPatientProfile')}`}
            accessibilityRole="button"
            className="app-card flex-row items-center active:opacity-80"
            key={patient.id}
            onPress={() => router.push(`/clinician-patient/${patient.id}`)}>
            <View className="mr-4 h-14 w-14 items-center justify-center rounded-full bg-brand-soft">
              <Text className="text-base font-bold text-brand-dark">{patient.initials}</Text>
            </View>
            <View className="flex-1">
              <Text className="text-lg font-bold text-ink">{patient.name}</Text>
              <Text className="mt-1 text-sm text-copy">{t('timeInRangeShort', { percent: patient.timeInRange })} · {t('syncedToday')}</Text>
              <View className={`mt-2 self-start rounded-full px-3 py-1 ${patient.status === 'lowRangeReview' ? 'bg-status-low' : 'bg-status-high'}`}>
                <Text className={`text-sm font-semibold ${patient.status === 'lowRangeReview' ? 'text-on-secondary' : 'text-danger'}`}>
                  {t(patient.status)}
                </Text>
              </View>
            </View>
            <Text className="ml-3 text-3xl font-light text-copy">›</Text>
          </Pressable>
        ))}
        {!patients.length ? <Text className="app-card py-8 text-center text-base text-copy">{t('noPatientsFound')}</Text> : null}
      </View>
    </SafeAreaView>
  );
}
