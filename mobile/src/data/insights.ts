import type { HistoryDay } from '@/data/history';

export type TimeInRange = {
  high: number;
  inRange: number;
  low: number;
};

export type PersonalInsights = {
  daysTracked: number;
  loggedMeals: number;
  responseMinutes: readonly number[];
  personalResponse: readonly number[];
  timeInRange: TimeInRange;
};

export const populationResponse = [0, 12, 24, 16, 6] as const;

export const featureImportance = [
  { label: 'Meal carbohydrates', value: 34 },
  { label: 'Baseline glucose trend', value: 22 },
  { label: 'Time of day', value: 18 },
  { label: 'Heart rate / activity', value: 14 },
  { label: 'Medication timing', value: 12 },
] as const;

export const responseMinutes = [0, 30, 60, 90, 120] as const;

function toPercent(value: number, total: number) {
  return Math.round((value / total) * 100);
}

export function getPersonalInsights(days: readonly HistoryDay[]): PersonalInsights {
  const glucoseReadings = days.flatMap((day) => day.glucose.map((point) => point.value));
  const low = glucoseReadings.filter((value) => value < 70).length;
  const inRange = glucoseReadings.filter((value) => value >= 70 && value <= 140).length;
  const high = glucoseReadings.filter((value) => value > 140).length;
  const responses = days.flatMap((day) => day.mealResponses);
  const personalResponse = responseMinutes.map((_, index) =>
    Math.round(responses.reduce((sum, response) => sum + response.deltas[index], 0) / responses.length),
  );

  return {
    daysTracked: days.length,
    loggedMeals: days.reduce((count, day) => count + day.meals.length, 0),
    personalResponse,
    responseMinutes,
    timeInRange: {
      low: toPercent(low, glucoseReadings.length),
      inRange: toPercent(inRange, glucoseReadings.length),
      high: toPercent(high, glucoseReadings.length),
    },
  };
}
