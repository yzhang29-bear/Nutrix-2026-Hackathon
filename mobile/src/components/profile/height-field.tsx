import { QuestionnaireField } from '@/components/profile/questionnaire-field';

type HeightFieldProps = {
  onChange: (height: string) => void;
  value: string;
};

function formatHeight(value: string, previousValue: string) {
  const enteredDigits = value.replace(/\D/g, '');
  const previousDigits = previousValue.replace(/\D/g, '');
  const removedFormattingCharacter = value.length < previousValue.length && enteredDigits === previousDigits;
  const digits = (removedFormattingCharacter ? previousDigits.slice(0, -1) : enteredDigits).slice(0, 3);

  if (!digits) {
    return '';
  }

  if (digits.length === 1) {
    return `${digits}'`;
  }

  return `${digits[0]}'${digits.slice(1)}\"`;
}

export function HeightField({ onChange, value }: HeightFieldProps) {
  return (
    <QuestionnaireField
      keyboardType="number-pad"
      label="Height"
      maxLength={5}
      onChangeText={(nextValue) => onChange(formatHeight(nextValue, value))}
      placeholder={'5\'5"'}
      suffix="ft / in"
      value={value}
    />
  );
}
