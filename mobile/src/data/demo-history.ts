import type { DemoPersonId } from '@/auth/types';
import type { HistoryDay, HistoryDayId } from '@/data/history';
import type { TimeInRange } from '@/data/insights';

type DemoHistory = {
  days: readonly HistoryDay[];
  latestDay: HistoryDay;
  participantLabel: string;
  response: readonly number[];
  responseSampleSize: number;
  sourceWindow: string;
  timeInRange: TimeInRange;
  trackedMeals: number;
};

const dayIds: readonly HistoryDayId[] = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'];

function dailyAverages(values: readonly number[]): readonly HistoryDay[] {
  return values.map((averageGlucose, index) => ({
    averageGlucose,
    events: [],
    glucose: [],
    heartRate: [],
    id: dayIds[index],
    mealResponses: [],
    meals: [],
  }));
}

const histories: Record<DemoPersonId, DemoHistory> = {
  'cgmacros-002': {
    participantLabel: 'CGMacros-002',
    sourceWindow: 'Nov 15–22, 2019',
    days: dailyAverages([67.2, 79.1, 70.8, 71.9, 89.4, 87.6, 80.8]),
    response: [0, 20.5, 34.9, 30.2, 26.7],
    responseSampleSize: 28,
    timeInRange: { low: 34, inRange: 64, high: 2 },
    trackedMeals: 28,
    latestDay: {
      id: 'D7',
      averageGlucose: 80.8,
      glucose: [
        { hour: 0, value: 136 }, { hour: 3, value: 70 }, { hour: 6, value: 78 }, { hour: 8, value: 74 },
        { hour: 10, value: 78 }, { hour: 12.5, value: 76 }, { hour: 15, value: 79 }, { hour: 18, value: 94 },
        { hour: 19.2, value: 81.4 }, { hour: 21, value: 65 }, { hour: 24, value: 99 },
      ],
      heartRate: [
        { hour: 0, value: 0 }, { hour: 3, value: 73 }, { hour: 6, value: 74 }, { hour: 8, value: 69 },
        { hour: 10, value: 72 }, { hour: 12.5, value: 95 }, { hour: 15, value: 75 }, { hour: 18, value: 73 },
        { hour: 19.2, value: 71 }, { hour: 21, value: 66 }, { hour: 24, value: 70 },
      ],
      events: [
        { hour: 12.6, kind: 'meal', value: 78.8 }, { hour: 17.07, kind: 'meal', value: 60.3 }, { hour: 22.63, kind: 'meal', value: 95.4 },
      ],
      meals: [
        { name: 'Breakfast', summary: 'Recorded meal · 608 kcal', time: '8:48 AM', carbs: 66, peak: 112.6 },
        { name: 'Lunch', summary: 'Recorded meal · 585 kcal', time: '1:16 PM', carbs: 40, peak: 92.7 },
        { name: 'Dinner', summary: 'Recorded meal · 626 kcal', time: '6:50 PM', carbs: 42, peak: 98.7 },
      ],
      mealResponses: [{ deltas: [0, 33.8, 8.2, 15.2, 15.4] }, { deltas: [0, 21.7, 32.4, 26.7, 22.2] }, { deltas: [0, 3.3, -4.3, 2.5, -9.1] }],
    },
  },
  'cgmacros-003': {
    participantLabel: 'CGMacros-003',
    sourceWindow: 'Mar 11–18, 2020',
    days: dailyAverages([95.2, 110.3, 108.2, 116.4, 103.9, 106.2, 112]),
    response: [0, 38.6, 63.4, 46.1, 24.3],
    responseSampleSize: 22,
    timeInRange: { low: 2, inRange: 84, high: 14 },
    trackedMeals: 22,
    latestDay: {
      id: 'D7',
      averageGlucose: 112,
      glucose: [
        { hour: 0, value: 111 }, { hour: 3, value: 75 }, { hour: 6, value: 181 }, { hour: 8, value: 109 },
        { hour: 10, value: 88 }, { hour: 12.5, value: 166 }, { hour: 15, value: 106 }, { hour: 18, value: 89 },
        { hour: 19.2, value: 86 }, { hour: 21, value: 89 }, { hour: 24, value: 108 },
      ],
      heartRate: [
        { hour: 0, value: 85 }, { hour: 3, value: 135 }, { hour: 6, value: 84 }, { hour: 8, value: 85 },
        { hour: 10, value: 79 }, { hour: 12.5, value: 84 }, { hour: 15, value: 80 }, { hour: 18, value: 79 },
        { hour: 19.2, value: 79 }, { hour: 21, value: 80 }, { hour: 24, value: 81 },
      ],
      events: [
        { hour: 0.07, kind: 'meal', value: 118.7 }, { hour: 4.35, kind: 'meal', value: 78 }, { hour: 10.98, kind: 'meal', value: 85.7 },
      ],
      meals: [
        { name: 'Breakfast', summary: 'Recorded meal · 448 kcal', time: '9:07 AM', carbs: 66, peak: 228.8 },
        { name: 'Lunch', summary: 'Recorded meal · 575 kcal', time: '1:24 PM', carbs: 76, peak: 177.4 },
        { name: 'Dinner', summary: 'Recorded meal · 690 kcal', time: '8:02 PM', carbs: 89, peak: 166.1 },
      ],
      mealResponses: [{ deltas: [0, 82.7, 110.1, 45.1, -4.9] }, { deltas: [0, 31.4, 73, 99.4, 83.2] }, { deltas: [0, 64.6, 76.5, 80.3, 41.7] }],
    },
  },
};

type TimelineSample = readonly [number, number, number];

// [hours since day start, glucose mg/dL, heart rate bpm], sampled from each recorded day.
const timelineSamples: Record<DemoPersonId, readonly (readonly TimelineSample[])[]> = {
  'cgmacros-002': [
    [[0, 56, 82], [6, 68, 66], [12.5, 71, 118], [18, 64, 77], [24, 94, 77]],
    [[0, 94, 77], [6, 86, 70], [12.5, 78, 78], [18, 62, 92], [24, 79, 79]],
    [[0, 79, 79], [6, 53, 69], [12.5, 56, 85], [18, 87, 104], [24, 65, 92]],
    [[0, 65, 92], [6, 68, 72], [12.5, 82, 107], [18, 67, 88], [24, 143, 87]],
    [[0, 143, 87], [6, 78, 76], [12.5, 91, 91], [18, 129, 94], [24, 82, 102]],
    [[0, 82, 102], [6, 76, 68], [12.5, 78, 76], [18, 52, 96], [24, 136, 80]],
  ],
  'cgmacros-003': [
    [[0, 155, 78], [6, 76, 80], [12.5, 159, 86], [18, 74, 80], [24, 93, 79]],
    [[0, 93, 79], [6, 115, 94], [12.5, 164, 83], [18, 89, 80], [24, 217, 84]],
    [[0, 217, 84], [6, 102, 90], [12.5, 150, 84], [18, 91, 80], [24, 103, 79]],
    [[0, 103, 79], [6, 216, 82], [12.5, 175, 110], [18, 91, 80], [24, 106, 91]],
    [[0, 106, 91], [6, 88, 80], [12.5, 131, 84], [18, 84, 79], [24, 208, 85]],
    [[0, 208, 85], [6, 149, 88], [12.5, 121, 91], [18, 118, 79], [24, 111, 85]],
  ],
};

function buildDetailedDays(personId: DemoPersonId, history: DemoHistory): readonly HistoryDay[] {
  return history.days.map((day, index) => {
    if (index === 6) return history.latestDay;

    const samples = timelineSamples[personId][index];
    return {
      ...day,
      glucose: samples.map(([hour, value]) => ({ hour, value })),
      heartRate: samples.map(([hour, , value]) => ({ hour, value })),
    };
  });
}

export function getDemoHistory(personId?: DemoPersonId): DemoHistory {
  const resolvedPersonId = personId ?? 'cgmacros-003';
  const history = histories[resolvedPersonId];
  return { ...history, days: buildDetailedDays(resolvedPersonId, history) };
}
