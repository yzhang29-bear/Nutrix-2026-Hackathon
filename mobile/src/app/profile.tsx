import { useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';

import { QuestionnaireField } from '@/components/profile/questionnaire-field';
import { HeightField } from '@/components/profile/height-field';
import { SelectionChip } from '@/components/profile/selection-chip';
import {
  diabetesTypes,
  ethnicities,
  genders,
  managementOptions,
  useHealthQuestionnaire,
  type Ethnicity,
  type ManagementOption,
} from '@/features/profile/health-questionnaire-context';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

function toggleItem<T extends string>(items: T[], value: T) {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

function ProfileValue({ label, value }: { label: string; value?: string }) {
  const { colors, text } = usePatientTheme();
  return (
    <View className="min-w-[45%] flex-1 rounded-card bg-canvas p-4" style={{ backgroundColor: colors.background }}>
      <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{label}</Text>
      <Text className="mt-1 text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{value || 'Not provided'}</Text>
    </View>
  );
}

export default function ProfileScreen() {
  const router = useRouter();
  const { answers, updateAnswers } = useHealthQuestionnaire();
  const { t } = useLanguage();
  const { cardStyle, colors, controlStyle, primaryButtonStyle, text } = usePatientTheme();
  const [editing, setEditing] = useState(false);

  function save() {
    setEditing(false);
  }

  return (
    <SafeAreaView className="app-screen" edges={['top']} style={{ backgroundColor: colors.background }}>
      <ScrollView contentContainerClassName="px-5 pb-12" keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        <View className="flex-row items-center py-5">
          <Pressable
            accessibilityLabel={t('backToSettings')}
            accessibilityRole="button"
            className="mr-3 h-10 w-10 items-center justify-center rounded-full active:bg-line"
            onPress={() => router.replace('/(tabs)/settings')}
            style={controlStyle}>
            <Text className="-mt-1 text-4xl font-light leading-none text-ink" style={{ color: colors.foreground }}>‹</Text>
          </Pressable>
          <Text className="text-3xl font-bold tracking-tight text-ink" style={[text.pageTitle, { color: colors.foreground }]}>{t('healthProfile')}</Text>
        </View>

        {!editing ? (
          <>
            <View className="app-card" style={cardStyle}>
              <View className="flex-row items-center justify-between">
                <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('aboutYou')}</Text>
                <Text className="rounded-full bg-brand-soft px-3 py-1 text-xs font-bold text-brand-dark" style={[text.caption, { backgroundColor: colors.primarySoft, color: colors.onPrimarySoft }]}>{t('sessionOnly')}</Text>
              </View>
              <View className="mt-5 flex-row flex-wrap gap-3">
                <ProfileValue label="Age" value={answers.age ? `${answers.age} years` : undefined} />
                <ProfileValue label="Height" value={answers.height} />
                <ProfileValue label="Weight" value={answers.weight ? `${answers.weight} lb` : undefined} />
                <ProfileValue label="Gender" value={answers.gender} />
                {answers.gender === 'Female' ? <ProfileValue label="Pregnancy" value={answers.isPregnant} /> : null}
              </View>
              <Text className="mt-5 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('ethnicity')}</Text>
              <Text className="mt-1 text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{answers.ethnicity.join(', ') || t('notProvided')}</Text>
            </View>

            <View className="app-card mt-4" style={cardStyle}>
              <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('diabetesHistory')}</Text>
              <View className="mt-5 flex-row flex-wrap gap-3">
                <ProfileValue label="Diabetes type" value={answers.diabetesType} />
              </View>
              <Text className="mt-5 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('management')}</Text>
              <Text className="mt-1 text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{answers.diabetesManagement.join(', ') || t('notProvided')}</Text>
            </View>

            <Pressable accessibilityRole="button" className="app-button-primary mt-5" onPress={() => setEditing(true)} style={primaryButtonStyle}>
              <Text className="text-base font-bold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>{t('editHealthProfile')}</Text>
            </Pressable>
          </>
        ) : (
          <>
            <View className="app-card" style={cardStyle}>
              <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('aboutYou')}</Text>
              <View className="mt-5">
                <QuestionnaireField keyboardType="number-pad" label={t('age')} onChangeText={(age) => updateAnswers({ age })} placeholder="42" suffix="years" value={answers.age} />
              </View>
              <View className="mt-5 flex-row gap-4">
                <HeightField onChange={(height) => updateAnswers({ height })} value={answers.height} />
                <QuestionnaireField keyboardType="decimal-pad" label={t('weight')} onChangeText={(weight) => updateAnswers({ weight })} placeholder="150" suffix="lb" value={answers.weight} />
              </View>
              <Text className="mt-6 text-base font-semibold text-ink">{t('gender')}</Text>
              <View className="mt-3 flex-row flex-wrap gap-2">
                {genders.map((gender) => (
                  <SelectionChip
                    key={gender}
                    label={gender}
                    onPress={() => updateAnswers({ gender, isPregnant: gender === 'Female' ? answers.isPregnant : undefined })}
                    selected={answers.gender === gender}
                  />
                ))}
              </View>
              {answers.gender === 'Female' ? (
                <View className="mt-4 w-full rounded-card bg-brand-soft p-4" style={{ backgroundColor: colors.primarySoft }}>
                  <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{t('pregnancy')}</Text>
                  <View className="mt-3 flex-row flex-wrap gap-2">
                    {(['Yes', 'No', 'Not sure'] as const).map((status) => (
                      <SelectionChip key={status} label={status} onPress={() => updateAnswers({ isPregnant: status })} selected={answers.isPregnant === status} />
                    ))}
                  </View>
                </View>
              ) : null}
              <Text className="mt-6 text-base font-semibold text-ink">{t('ethnicity')}</Text>
              <Text className="mt-1 text-sm text-copy">{t('selectAll')}</Text>
              <View className="mt-3 flex-row flex-wrap gap-2">
                {ethnicities.map((ethnicity) => (
                  <SelectionChip
                    key={ethnicity}
                    label={ethnicity}
                    onPress={() => updateAnswers({ ethnicity: toggleItem(answers.ethnicity, ethnicity as Ethnicity) })}
                    selected={answers.ethnicity.includes(ethnicity as Ethnicity)}
                  />
                ))}
              </View>
            </View>

            <View className="app-card mt-4" style={cardStyle}>
              <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('diabetesHistory')}</Text>
              <Text className="mt-5 text-base font-semibold text-ink">{t('diabetesType')}</Text>
              <View className="mt-3 flex-row flex-wrap gap-2">
                {diabetesTypes.map((type) => (
                  <SelectionChip key={type} label={type} onPress={() => updateAnswers({ diabetesType: type })} selected={answers.diabetesType === type} />
                ))}
              </View>
              <Text className="mt-6 text-base font-semibold text-ink">{t('diabetesManagement')}</Text>
              <View className="mt-3 flex-row flex-wrap gap-2">
                {managementOptions.map((option) => (
                  <SelectionChip
                    key={option}
                    label={option}
                    onPress={() => updateAnswers({ diabetesManagement: toggleItem(answers.diabetesManagement, option as ManagementOption) })}
                    selected={answers.diabetesManagement.includes(option as ManagementOption)}
                  />
                ))}
              </View>
            </View>

            <Pressable accessibilityRole="button" className="app-button-primary mt-5" onPress={save} style={primaryButtonStyle}>
              <Text className="text-base font-bold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>{t('saveChanges')}</Text>
            </Pressable>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
