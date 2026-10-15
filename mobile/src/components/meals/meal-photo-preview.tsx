import { Image, Pressable, Text, View } from 'react-native';

import type { MealPhoto } from '@/features/meals/meal-photo';
import type { PhotoPickerStatus } from '@/hooks/use-meal-photo-picker';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type MealPhotoPreviewProps = {
  onReplace: () => void;
  photo: MealPhoto;
  status: PhotoPickerStatus;
};

export function MealPhotoPreview({ onReplace, photo, status }: MealPhotoPreviewProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, controlStyle, text } = usePatientTheme();
  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>{t('photo')}</Text>
      <Text className="mt-1 text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>{t('nutritionEstimation')}</Text>
      <Image accessibilityLabel="Selected meal photo" className="mt-4 h-48 w-full rounded-card bg-line" source={{ uri: photo.uri }} />
      <Text className="mt-2 text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>
        JPEG · {photo.width} × {photo.height} · {(photo.byteSize / 1024 / 1024).toFixed(1)} MB
      </Text>
      <Pressable accessibilityRole="button" className="mt-4 self-start rounded-full px-4 py-2" onPress={onReplace} style={[controlStyle, { backgroundColor: colors.primarySoft }]}>
        <Text className="text-sm font-semibold text-brand-dark" style={[text.label, { color: colors.onPrimarySoft }]}>
          {status === 'processing' ? t('preparingPhoto') : t('replacePhoto')}
        </Text>
      </Pressable>
      <PhotoPickerStatusMessage status={status} />
    </View>
  );
}

export function PhotoPickerStatusMessage({ status }: { status: PhotoPickerStatus }) {
  const { t } = useLanguage();
  const { colors, text } = usePatientTheme();
  if (status === 'idle' || status === 'processing') return null;

  const message = {
    cancelled: t('photoCancelled'),
    'permission-denied': t('photoPermissionDenied'),
    'normalization-error': t('photoNormalizationError'),
    oversize: t('photoOversize'),
  }[status];

  return <Text className="mt-3 text-center text-sm" style={[text.label, { color: status === 'cancelled' ? colors.mutedForeground : colors.destructive }]}>{message}</Text>;
}
