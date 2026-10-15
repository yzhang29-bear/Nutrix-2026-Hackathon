import { Pressable, Text, View } from 'react-native';

import { mealTypes, type MealType } from '@/features/meals/meal-draft-context';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type MealTypeSelectorProps = {
  onChange: (mealType: MealType) => void;
  value: MealType;
};

export function MealTypeSelector({ onChange, value }: MealTypeSelectorProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();
  return (
    <View className="app-card" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('mealType')}</Text>
      <View className="mt-3 flex-row gap-2">
        {mealTypes.map((mealType) => {
          const selected = mealType === value;

          return (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected }}
              className="min-h-20 flex-1 items-center justify-center rounded-card border px-1"
              key={mealType}
              onPress={() => onChange(mealType)}
              style={[controlStyle, { backgroundColor: selected ? colors.primarySoft : colors.background, borderColor: selected ? colors.primary : 'transparent' }]}>
              <Text className="text-center font-semibold" style={[text.label, { color: selected ? colors.onPrimarySoft : colors.foreground }]}>
                {t(mealType)}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}
