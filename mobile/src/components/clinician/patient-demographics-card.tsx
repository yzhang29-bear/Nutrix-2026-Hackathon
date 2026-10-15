import { Text, View } from 'react-native';

import type { ClinicianPatient } from '@/data/clinician-patients';

type PatientDemographicsCardProps = {
  patient: ClinicianPatient;
  t: (key: string, values?: Record<string, string | number>) => string;
};

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <View className="min-w-[30%] flex-1 rounded-card bg-canvas px-3 py-3">
      <Text className="text-xs text-copy">{label}</Text>
      <Text className="mt-1 text-base font-bold text-ink">{value}</Text>
    </View>
  );
}

export function PatientDemographicsCard({ patient, t }: PatientDemographicsCardProps) {
  const { demographics } = patient;

  return (
    <View className="app-card">
      <View className="flex-row items-center">
        <View className="mr-4 h-14 w-14 items-center justify-center rounded-full bg-brand-soft">
          <Text className="text-base font-bold text-brand-dark">{patient.initials}</Text>
        </View>
        <View className="flex-1">
          <Text className="text-xl font-bold text-ink">{patient.name}</Text>
          <Text className="mt-1 text-sm text-copy">
            {t(demographics.gender)} · {t(demographics.diabetesType)} · {t('yearsOld', { age: demographics.age })}
          </Text>
        </View>
      </View>

      <View className="mt-5 flex-row flex-wrap gap-2">
        <Detail label={t('height')} value={demographics.height} />
        <Detail label={t('weight')} value={demographics.weight} />
        <Detail label={t('bmi')} value={String(demographics.bmi)} />
        <Detail label={t('ethnicity')} value={t(demographics.ethnicity)} />
        <Detail label={t('pregnant')} value={demographics.isPregnant ? t(demographics.isPregnant) : t('notProvided')} />
        <Detail label={t('hba1c')} value={`${demographics.hba1c}%`} />
      </View>
    </View>
  );
}
