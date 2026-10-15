import { useState } from 'react';
import { Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

import { MealPhotoError, normalizeMealPhoto, type MealPhoto } from '@/features/meals/meal-photo';

export type PhotoPickerStatus =
  | 'cancelled'
  | 'idle'
  | 'normalization-error'
  | 'oversize'
  | 'permission-denied'
  | 'processing';

type UseMealPhotoPickerOptions = {
  onPhotoReady: (photo: MealPhoto) => void | Promise<void>;
};

export function useMealPhotoPicker({ onPhotoReady }: UseMealPhotoPickerOptions) {
  const [status, setStatus] = useState<PhotoPickerStatus>('idle');

  async function processPickerResult(result: ImagePicker.ImagePickerResult) {
    if (result.canceled) {
      setStatus('cancelled');
      return;
    }

    setStatus('processing');

    try {
      const photo = await normalizeMealPhoto(result.assets[0]);
      await onPhotoReady(photo);
      setStatus('idle');
    } catch (error) {
      setStatus(error instanceof MealPhotoError && error.code === 'PHOTO_TOO_LARGE' ? 'oversize' : 'normalization-error');
    }
  }

  async function openCamera() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      setStatus('permission-denied');
      return;
    }

    try {
      await processPickerResult(await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 1 }));
    } catch {
      setStatus('normalization-error');
    }
  }

  async function openPhotoLibrary() {
    try {
      await processPickerResult(await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 }));
    } catch {
      setStatus('normalization-error');
    }
  }

  function choosePhotoSource() {
    if (status === 'processing') {
      return;
    }

    Alert.alert('Add meal photo', 'Choose a photo source.', [
      { text: 'Camera', onPress: () => void openCamera() },
      { text: 'Photo Library', onPress: () => void openPhotoLibrary() },
      { style: 'cancel', text: 'Cancel', onPress: () => setStatus('cancelled') },
    ]);
  }

  return { choosePhotoSource, status };
}
