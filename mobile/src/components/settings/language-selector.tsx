import { Modal, Pressable, Text, View } from 'react-native';

import { languages, type LanguageCode } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type LanguageSelectorProps = {
  isOpen: boolean;
  onClose: () => void;
  onOpen: () => void;
  onSelect: (language: LanguageCode) => void;
  selectedLanguage: LanguageCode;
  title: string;
  description: string;
};

export function LanguageSelector({ description, isOpen, onClose, onOpen, onSelect, selectedLanguage, title }: LanguageSelectorProps) {
  const selectedLabel = languages.find((language) => language.code === selectedLanguage)?.label;
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();

  return (
    <>
      <Pressable accessibilityRole="button" className="app-card mt-4 flex-row items-center justify-between" onPress={onOpen} style={[cardStyle, controlStyle]}>
        <View className="flex-1 pr-4">
          <Text className="text-base font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{title}</Text>
          <Text className="mt-1 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{description}</Text>
        </View>
        <View className="flex-row items-center">
          <Text className="mr-2 text-sm font-semibold text-brand-dark" style={[text.label, { color: colors.onPrimarySoft }]}>{selectedLabel}</Text>
          <Text className="text-2xl text-brand-dark" style={{ color: colors.onPrimarySoft }}>›</Text>
        </View>
      </Pressable>

      <Modal animationType="fade" onRequestClose={onClose} transparent visible={isOpen}>
        <Pressable accessibilityRole="button" className="flex-1 justify-end bg-black/30 p-5" onPress={onClose}>
          <Pressable accessibilityViewIsModal accessibilityRole="menu" className="rounded-card bg-surface p-2 shadow-card" onPress={(event) => event.stopPropagation()} style={{ backgroundColor: colors.card }}>
            <Text className="px-3 pb-2 pt-3 text-lg font-bold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>{title}</Text>
            {languages.map((language) => {
              const selected = language.code === selectedLanguage;

              return (
                <Pressable
                  key={language.code}
                  accessibilityRole="menuitem"
                  accessibilityState={{ selected }}
                  className="flex-row items-center justify-between rounded-card px-3 py-4"
                  onPress={() => {
                    onSelect(language.code);
                    onClose();
                  }}
                  style={[controlStyle, selected ? { backgroundColor: colors.primarySoft } : undefined]}>
                  <Text className={selected ? 'font-bold' : ''} style={[text.body, { color: selected ? colors.onPrimarySoft : colors.foreground }]}>{language.label}</Text>
                  {selected ? <Text className="text-lg font-bold text-brand-dark" style={{ color: colors.onPrimarySoft }}>✓</Text> : null}
                </Pressable>
              );
            })}
          </Pressable>
        </Pressable>
      </Modal>
    </>
  );
}
