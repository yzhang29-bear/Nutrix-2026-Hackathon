import * as ImageManipulator from 'expo-image-manipulator';
import type * as ImagePicker from 'expo-image-picker';

const maxImageEdge = 2048;
const maxImageBytes = 8 * 1024 * 1024;

export type MealPhoto = {
  byteSize: number;
  height: number;
  uri: string;
  width: number;
};

export class MealPhotoError extends Error {
  constructor(public readonly code: 'PHOTO_TOO_LARGE' | 'NORMALIZATION_FAILED') {
    super(code);
  }
}

function base64ByteSize(base64: string) {
  const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0;
  return Math.floor((base64.length * 3) / 4) - padding;
}

export async function normalizeMealPhoto(asset: ImagePicker.ImagePickerAsset): Promise<MealPhoto> {
  try {
    const context = ImageManipulator.ImageManipulator.manipulate(asset.uri);
    const longestEdge = Math.max(asset.width, asset.height);

    if (longestEdge > maxImageEdge) {
      context.resize(asset.width >= asset.height ? { width: maxImageEdge } : { height: maxImageEdge });
    }

    const renderedImage = await context.renderAsync();
    const output = await renderedImage.saveAsync({
      base64: true,
      compress: 0.8,
      format: ImageManipulator.SaveFormat.JPEG,
    });

    if (!output.base64) {
      throw new MealPhotoError('NORMALIZATION_FAILED');
    }

    const byteSize = base64ByteSize(output.base64);

    if (byteSize > maxImageBytes) {
      throw new MealPhotoError('PHOTO_TOO_LARGE');
    }

    return {
      byteSize,
      height: output.height,
      uri: output.uri,
      width: output.width,
    };
  } catch (error) {
    if (error instanceof MealPhotoError) {
      throw error;
    }

    throw new MealPhotoError('NORMALIZATION_FAILED');
  }
}
