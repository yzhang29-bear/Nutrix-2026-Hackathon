import { useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';

import { QuestionnaireField } from '@/components/profile/questionnaire-field';
import { HeightField } from '@/components/profile/height-field';
import { QuestionnaireProgress } from '@/components/profile/questionnaire-progress';
import { SelectionChip } from '@/components/profile/selection-chip';
import {
  diabetesTypes,
  ethnicities,
  genders,
  managementOptions,
  useHealthQuestionnaire,
  type DiabetesType,
  type Ethnicity,
  type Gender,
  type ManagementOption,
} from '@/features/profile/health-questionnaire-context';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

function toggleItem<T extends string>(items: T[], value: T) {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

export default function HealthQuestionnaireScreen() {
  const router = useRouter();
  const { answers, completeQuestionnaire, resetAnswers, updateAnswers } = useHealthQuestionnaire();
  const { t } = useLanguage();
  const { cardStyle, colors, controlStyle, primaryButtonStyle, text } = usePatientTheme();
  const [step, setStep] = useState<1 | 2>(1);

  function leaveQuestionnaire() {
    resetAnswers();
    completeQuestionnaire();
    router.replace('/(tabs)');
  }

  function finishQuestionnaire() {
    completeQuestionnaire();
    router.replace('/profile');
  }

  function selectGender(gender: Gender) {
    updateAnswers({ gender, isPregnant: gender === 'Female' ? answers.isPregnant : undefined });
  }

  if (step === 2) {
    return (
      <SafeAreaView className="app-screen" edges={['top']} style={{ backgroundColor: colors.background }}>
        <ScrollView contentContainerClassName="px-5 pb-12" showsVerticalScrollIndicator={false}>
          <View className="flex-row items-center justify-between pt-5">
            <Text className="text-2xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('healthQuestionnaire')}</Text>
            <Pressable accessibilityRole="button" onPress={leaveQuestionnaire}>
              <Text className="text-base font-bold text-brand-dark" style={[text.body, { color: colors.onPrimarySoft }]}>{t('skip')}</Text>
            </Pressable>
          </View>
          <QuestionnaireProgress step={2} />
          <Text className="mt-3 text-3xl font-bold tracking-tight text-ink" style={[text.pageTitle, { color: colors.foreground }]}>{t('diabetesHistory')}</Text>
          <Text className="app-body-copy mt-2" style={[text.body, { color: colors.mutedForeground }]}>{t('diabetesDescription')}</Text>

          <View className="app-card mt-6" style={cardStyle}>
            <View className="flex-row items-center">
              <View className="mr-3 h-8 w-8 items-center justify-center rounded-full bg-brand-soft" style={{ backgroundColor: colors.primarySoft }}>
                <Text className="font-bold text-brand-dark" style={{ color: colors.onPrimarySoft }}>2</Text>
              </View>
              <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('diabetesHistory')}</Text>
            </View>
            <Text className="mt-6 text-base font-semibold text-ink">{t('diabetesType')}</Text>
            <View className="mt-3 flex-row flex-wrap gap-2">
              {diabetesTypes.map((type) => (
                <SelectionChip
                  key={type}
                  label={type}
                  onPress={() => updateAnswers({ diabetesType: type as DiabetesType })}
                  selected={answers.diabetesType === type}
                />
              ))}
            </View>

            <Text className="mt-7 text-base font-semibold text-ink">{t('diabetesManagement')}</Text>
            <Text className="mt-1 text-sm text-copy">{t('selectAll')}</Text>
            <View className="mt-3 flex-row flex-wrap gap-2">
              {managementOptions.map((option) => (
                <SelectionChip
                  key={option}
                  label={option}
                  onPress={() =>
                    updateAnswers({
                      diabetesManagement: toggleItem(answers.diabetesManagement, option as ManagementOption),
                    })
                  }
                  selected={answers.diabetesManagement.includes(option as ManagementOption)}
                />
              ))}
            </View>
          </View>

          <View className="mt-5 flex-row gap-3">
            <Pressable accessibilityRole="button" className="flex-1 items-center justify-center rounded-card border border-brand px-4 py-4" onPress={() => setStep(1)} style={[controlStyle, { borderColor: colors.primary }] }>
              <Text className="text-base font-bold text-brand-dark" style={[text.button, { color: colors.onPrimarySoft }]}>{t('back')}</Text>
            </Pressable>
            <Pressable accessibilityRole="button" className="app-button-primary flex-1" onPress={finishQuestionnaire} style={primaryButtonStyle}>
              <Text className="text-base font-bold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>{t('finish')}</Text>
            </Pressable>
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView className="app-screen" edges={['top']} style={{ backgroundColor: colors.background }}>
      <ScrollView contentContainerClassName="px-5 pb-12" keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        <View className="flex-row items-center justify-between pt-5">
          <Text className="text-2xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('healthQuestionnaire')}</Text>
          <Pressable accessibilityRole="button" onPress={leaveQuestionnaire}>
            <Text className="text-base font-bold text-brand-dark" style={[text.body, { color: colors.onPrimarySoft }]}>{t('skip')}</Text>
          </Pressable>
        </View>
        <QuestionnaireProgress step={1} />
        <Text className="mt-3 text-3xl font-bold tracking-tight text-ink" style={[text.pageTitle, { color: colors.foreground }]}>{t('tellUs')}</Text>
        <Text className="app-body-copy mt-2" style={[text.body, { color: colors.mutedForeground }]}>
          {t('questionnaireDescription')}
        </Text>

        <View className="app-card mt-6" style={cardStyle}>
          <View className="flex-row items-center">
            <View className="mr-3 h-8 w-8 items-center justify-center rounded-full bg-brand-soft" style={{ backgroundColor: colors.primarySoft }}>
              <Text className="font-bold text-brand-dark" style={{ color: colors.onPrimarySoft }}>1</Text>
            </View>
            <Text className="text-xl font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{t('aboutYou')}</Text>
          </View>

          <View className="mt-6">
            <QuestionnaireField
              keyboardType="number-pad"
              label={t('age')}
              onChangeText={(age) => updateAnswers({ age })}
              placeholder="42"
              suffix="years"
              value={answers.age}
            />
          </View>
          <View className="mt-5 flex-row gap-4">
            <HeightField onChange={(height) => updateAnswers({ height })} value={answers.height} />
            <QuestionnaireField
              keyboardType="decimal-pad"
              label={t('weight')}
              onChangeText={(weight) => updateAnswers({ weight })}
              placeholder="150"
              suffix="lb"
              value={answers.weight}
            />
          </View>

          <Text className="mt-7 text-base font-semibold text-ink">{t('gender')}</Text>
          <View className="mt-3 flex-row flex-wrap gap-2">
            {genders.map((gender) => (
              <SelectionChip key={gender} label={gender} onPress={() => selectGender(gender)} selected={answers.gender === gender} />
            ))}
          </View>

          {answers.gender === 'Female' ? (
            <View className="mt-4 w-full self-stretch rounded-card bg-brand-soft p-4" style={{ backgroundColor: colors.primarySoft }}>
              <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{t('pregnancy')}</Text>
              <Text className="mt-1 text-sm leading-5 text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('pregnancyDescription')}</Text>
              <View className="mt-4 flex-row flex-wrap gap-2">
                {(['Yes', 'No', 'Not sure'] as const).map((status) => (
                  <SelectionChip
                    key={status}
                    label={status}
                    onPress={() => updateAnswers({ isPregnant: status })}
                    selected={answers.isPregnant === status}
                  />
                ))}
              </View>
            </View>
          ) : null}

          <Text className="mt-7 text-base font-semibold text-ink">{t('ethnicity')}</Text>
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

        <Pressable accessibilityRole="button" className="app-button-primary mt-5" onPress={() => setStep(2)} style={primaryButtonStyle}>
          <Text className="text-base font-bold text-on-brand" style={[text.button, { color: colors.onPrimary }]}>{t('continue')}</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}
