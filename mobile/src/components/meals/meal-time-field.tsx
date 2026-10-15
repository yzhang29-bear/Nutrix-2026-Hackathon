import { useState } from 'react';
import { Modal, Platform, Pressable, Text, View } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type MealTimeFieldProps = {
  onChange: (time: Date) => void;
  value: Date;
};

const demoMealTimes = [
  { label: 'Demo 003 lunch · Mar 18, 2020, 1:34 PM', time: new Date(2020, 2, 18, 13, 34) },
  { label: 'Demo 002 lunch · Nov 23, 2019, 2:30 PM', time: new Date(2019, 10, 23, 14, 30) },
] as const;

function formatMealTime(time: Date) {
  return time.toLocaleString([], {
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

export function MealTimeField({ onChange, value }: MealTimeFieldProps) {
  const [pickerVisible, setPickerVisible] = useState(false);
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();

  function handleChange(_: unknown, selectedTime?: Date) {
    if (selectedTime) {
      onChange(selectedTime);
    }

    if (Platform.OS === 'android') {
      setPickerVisible(false);
    }
  }

  return (
    <View className="app-card flex-1" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>Meal date & time</Text>
      <Pressable accessibilityRole="button" className="mt-3 rounded-card px-4 py-4" onPress={() => setPickerVisible(true)} style={[controlStyle, { backgroundColor: colors.background }]}>
        <Text className="font-semibold text-ink" style={[text.body, { color: colors.foreground }]}>{formatMealTime(value)}</Text>
      </Pressable>
      <View className="mt-2 gap-1">
        {demoMealTimes.map((demo) => (
          <Pressable
            accessibilityLabel={`Use ${demo.label}`}
            accessibilityRole="button"
            className="rounded-lg px-1 py-1 active:bg-line"
            key={demo.label}
            onPress={() => onChange(demo.time)}>
            <Text className="font-semibold text-brand-dark" style={[text.caption, { color: colors.onPrimarySoft }]}>{demo.label}</Text>
          </Pressable>
        ))}
      </View>
      <Modal animationType="slide" onRequestClose={() => setPickerVisible(false)} transparent visible={pickerVisible && Platform.OS === 'ios'}>
        <View className="flex-1 justify-end bg-black/30">
          <View className="rounded-t-[28px] bg-surface p-5" style={{ backgroundColor: colors.card }}>
            <View className="flex-row items-center justify-between">
              <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>Meal date & time</Text>
              <Pressable accessibilityRole="button" onPress={() => setPickerVisible(false)}>
                <Text className="font-bold text-brand-dark" style={[text.body, { color: colors.onPrimarySoft }]}>Done</Text>
              </Pressable>
            </View>
            <DateTimePicker display="spinner" mode="datetime" onChange={handleChange} value={value} />
          </View>
        </View>
      </Modal>
      {pickerVisible && Platform.OS === 'android' ? (
        <DateTimePicker display="default" mode="datetime" onChange={handleChange} value={value} />
      ) : null}
    </View>
  );
}
