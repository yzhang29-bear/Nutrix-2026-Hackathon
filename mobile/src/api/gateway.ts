import { fetch as expoFetch } from 'expo/fetch';
import { File } from 'expo-file-system';

const gatewayUrl = process.env.EXPO_PUBLIC_NUTRIX_API_URL?.replace(/\/$/, '');

export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';

export type Readiness = {
  daysCollected: number;
  requiredDays: number;
  status: 'collecting_baseline' | 'needs_recent_cgm' | 'processing_baseline' | 'ready';
};

export type PhotoUpload = {
  headers: Record<string, string>;
  photoKey: string;
  uploadUrl: string;
};

export type Prediction = {
  confidence: { flags: string[]; level: 'high' | 'low' | 'medium' };
  explanation: { contributions: { factor: string; mgdl: number }[] };
  meal: {
    estimated_nutrition: { carbs_g: number; carbs_uncertainty_g: number; fat_g: number; fiber_g: number; kcal: number; protein_g: number };
    meal_type: MealType;
    photo_tags: { label: string; p: number }[];
    time: string;
  };
  observed?: {
    curve: { glucose_mgdl: number[]; minutes: number[] };
    glucose_at_min: Record<string, number>;
    iauc_0_120_mgdl_min: number;
    peak_glucose_mgdl: number;
    time_to_peak_min: number;
  };
  prediction: {
    curve: { glucose_mgdl: number[]; lower80: number[]; minutes: number[]; upper80: number[] };
    glucose_at_min: Record<string, number>;
    iauc_0_120_mgdl_min: number;
    peak_glucose_mgdl: number;
    peak_rise_mgdl: number;
    time_to_peak_min: number;
  };
  pre_meal_state: { glucose_mgdl: number };
};

export class GatewayError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
  }
}

type GatewayClientOptions = {
  getToken: () => Promise<string | null>;
};

async function errorMessage(response: Response) {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') return body.detail;
    if (body && typeof body === 'object' && 'message' in body && typeof body.message === 'string') return body.message;
  } catch {
    // A non-JSON error response still has a useful status below.
  }
  return `Request failed (${response.status}).`;
}

export function createGatewayClient({ getToken }: GatewayClientOptions) {
  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    if (!gatewayUrl) {
      throw new GatewayError('The Nutrix service is not configured. Add EXPO_PUBLIC_NUTRIX_API_URL to mobile/.env.');
    }

    const token = await getToken();
    if (!token) throw new GatewayError('Your session has expired. Please sign in again.', 401);

    let response: Response;
    try {
      response = await fetch(`${gatewayUrl}${path}`, {
        ...init,
        headers: { Accept: 'application/json', Authorization: `Bearer ${token}`, ...init?.headers },
      });
    } catch {
      throw new GatewayError('Unable to reach Nutrix. Check your connection and try again.');
    }

    if (!response.ok) throw new GatewayError(await errorMessage(response), response.status);
    return response.json() as Promise<T>;
  }

  return {
    getReadiness: () => request<Readiness>('/v1/mobile/readiness'),
    createPhotoUpload: () => request<PhotoUpload>('/v1/mobile/photos/upload-url', { method: 'POST' }),
    requestPrediction: (input: { logMeal: boolean; mealTime: string; mealType: MealType; photoKey: string }) =>
      request<Prediction>('/v1/mobile/predictions', {
        body: JSON.stringify({
          log_meal: input.logMeal,
          meal_time: input.mealTime,
          meal_type: input.mealType,
          photo_key: input.photoKey,
        }),
        headers: { 'Content-Type': 'application/json' },
        method: 'POST',
      }),
  };
}

export async function uploadMealPhoto(photoUri: string, upload: PhotoUpload) {
  try {
    // Sending the local file directly avoids React Native's legacy Blob bridge,
    // which can fail for ImagePicker's cached iOS assets.
    const photo = new File(photoUri);
    if (!photo.exists || photo.size === 0) throw new Error('photo file is unavailable');

    const response = await expoFetch(upload.uploadUrl, {
      body: photo,
      headers: upload.headers,
      method: 'PUT',
    });

    if (!response.ok) {
      throw new GatewayError(`Photo upload was rejected by storage (${response.status}).`, response.status);
    }
  } catch (error) {
    if (error instanceof GatewayError) throw error;
    throw new GatewayError('Photo upload failed. Check your connection and try again.');
  }
}
