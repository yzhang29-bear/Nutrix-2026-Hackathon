export type HistoryDayId = 'D1' | 'D2' | 'D3' | 'D4' | 'D5' | 'D6' | 'D7';

export type HistorySeriesPoint = {
  hour: number;
  value: number;
};

export type HistoryEvent = {
  hour: number;
  kind: 'meal' | 'medication';
  value: number;
};

export type LoggedMeal = {
  carbs: number;
  name: string;
  peak: number;
  summary: string;
  time: string;
};

export type MealResponse = {
  deltas: readonly number[];
};

export type HistoryDay = {
  averageGlucose: number;
  events: readonly HistoryEvent[];
  glucose: readonly HistorySeriesPoint[];
  heartRate: readonly HistorySeriesPoint[];
  id: HistoryDayId;
  meals: readonly LoggedMeal[];
  mealResponses: readonly MealResponse[];
};

export const historyDayOrder: readonly HistoryDayId[] = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'];

const hours = [0, 3, 6, 8, 10, 12.5, 15, 18, 19.2, 21, 24] as const;

function series(values: readonly number[]): HistorySeriesPoint[] {
  return hours.map((hour, index) => ({ hour, value: values[index] }));
}

function mealResponses(...responses: readonly number[][]): readonly MealResponse[] {
  return responses.map((deltas) => ({ deltas }));
}

export const historyByDay: Record<HistoryDayId, HistoryDay> = {
  D1: {
    id: 'D1',
    averageGlucose: 112,
    glucose: series([98, 92, 101, 118, 134, 112, 128, 118, 143, 121, 110]),
    heartRate: series([78, 74, 76, 88, 106, 94, 112, 91, 118, 89, 80]),
    events: [
      { hour: 7.8, kind: 'meal', value: 116 },
      { hour: 13, kind: 'meal', value: 113 },
      { hour: 19.3, kind: 'meal', value: 141 },
      { hour: 8, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Yogurt and granola', time: '7:45 AM', carbs: 42, peak: 136 },
      { name: 'Lunch', summary: 'Turkey sandwich', time: '12:35 PM', carbs: 51, peak: 144 },
      { name: 'Dinner', summary: 'Salmon and rice', time: '7:20 PM', carbs: 58, peak: 151 },
    ],
    mealResponses: mealResponses([0, 15, 31, 23, 7], [0, 17, 35, 25, 8], [0, 20, 40, 29, 10]),
  },
  D2: {
    id: 'D2',
    averageGlucose: 118,
    glucose: series([102, 98, 105, 122, 145, 119, 138, 124, 149, 129, 114]),
    heartRate: series([76, 75, 78, 91, 121, 98, 116, 95, 125, 94, 79]),
    events: [
      { hour: 7.6, kind: 'meal', value: 121 },
      { hour: 12.4, kind: 'meal', value: 120 },
      { hour: 19.1, kind: 'meal', value: 147 },
      { hour: 8.1, kind: 'medication', value: 66 },
      { hour: 20.2, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Eggs and toast', time: '7:30 AM', carbs: 37, peak: 142 },
      { name: 'Lunch', summary: 'Lentil soup', time: '12:20 PM', carbs: 46, peak: 147 },
      { name: 'Dinner', summary: 'Taco bowl', time: '7:05 PM', carbs: 64, peak: 158 },
    ],
    mealResponses: mealResponses([0, 18, 36, 26, 8], [0, 16, 33, 24, 7], [0, 22, 43, 31, 11]),
  },
  D3: {
    id: 'D3',
    averageGlucose: 109,
    glucose: series([95, 91, 97, 110, 129, 106, 119, 108, 135, 114, 103]),
    heartRate: series([74, 71, 75, 86, 104, 88, 101, 84, 109, 86, 77]),
    events: [
      { hour: 7.9, kind: 'meal', value: 109 },
      { hour: 12.7, kind: 'meal', value: 106 },
      { hour: 19, kind: 'meal', value: 133 },
      { hour: 8, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Oatmeal with berries', time: '7:50 AM', carbs: 48, peak: 130 },
      { name: 'Lunch', summary: 'Chicken salad', time: '12:40 PM', carbs: 29, peak: 122 },
      { name: 'Dinner', summary: 'Vegetable stir-fry', time: '6:55 PM', carbs: 54, peak: 139 },
    ],
    mealResponses: mealResponses([0, 14, 28, 20, 6], [0, 11, 23, 17, 5], [0, 17, 34, 24, 8]),
  },
  D4: {
    id: 'D4',
    averageGlucose: 121,
    glucose: series([104, 99, 108, 126, 151, 124, 143, 130, 156, 132, 118]),
    heartRate: series([79, 76, 82, 96, 126, 102, 121, 99, 129, 98, 84]),
    events: [
      { hour: 7.7, kind: 'meal', value: 125 },
      { hour: 12.6, kind: 'meal', value: 124 },
      { hour: 19.2, kind: 'meal', value: 154 },
      { hour: 8.1, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Bagel and cream cheese', time: '7:40 AM', carbs: 61, peak: 152 },
      { name: 'Lunch', summary: 'Grain bowl', time: '12:35 PM', carbs: 55, peak: 149 },
      { name: 'Dinner', summary: 'Sushi rolls', time: '7:10 PM', carbs: 72, peak: 163 },
    ],
    mealResponses: mealResponses([0, 23, 44, 32, 11], [0, 19, 38, 28, 9], [0, 26, 49, 35, 13]),
  },
  D5: {
    id: 'D5',
    averageGlucose: 115,
    glucose: series([100, 94, 103, 119, 140, 115, 132, 119, 146, 124, 111]),
    heartRate: series([77, 73, 79, 92, 116, 95, 111, 91, 119, 90, 80]),
    events: [
      { hour: 7.5, kind: 'meal', value: 118 },
      { hour: 12.4, kind: 'meal', value: 114 },
      { hour: 19.2, kind: 'meal', value: 144 },
      { hour: 8, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Peanut butter toast', time: '7:30 AM', carbs: 44, peak: 141 },
      { name: 'Lunch', summary: 'Quinoa bowl', time: '12:20 PM', carbs: 52, peak: 136 },
      { name: 'Dinner', summary: 'Pasta primavera', time: '7:10 PM', carbs: 66, peak: 154 },
    ],
    mealResponses: mealResponses([0, 17, 34, 25, 8], [0, 16, 31, 22, 7], [0, 21, 41, 30, 10]),
  },
  D6: {
    id: 'D6',
    averageGlucose: 124,
    glucose: series([108, 104, 112, 130, 154, 128, 149, 134, 162, 138, 121]),
    heartRate: series([80, 77, 84, 99, 128, 105, 124, 104, 132, 101, 86]),
    events: [
      { hour: 7.8, kind: 'meal', value: 129 },
      { hour: 12.5, kind: 'meal', value: 127 },
      { hour: 19.3, kind: 'meal', value: 160 },
      { hour: 8, kind: 'medication', value: 66 },
      { hour: 20.1, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Pancakes and fruit', time: '7:45 AM', carbs: 68, peak: 157 },
      { name: 'Lunch', summary: 'Chicken rice bowl', time: '12:30 PM', carbs: 62, peak: 155 },
      { name: 'Dinner', summary: 'Burger and sweet potato', time: '7:15 PM', carbs: 74, peak: 166 },
    ],
    mealResponses: mealResponses([0, 25, 48, 35, 12], [0, 22, 43, 31, 10], [0, 29, 54, 39, 14]),
  },
  D7: {
    id: 'D7',
    averageGlucose: 115,
    glucose: series([101, 93, 102, 117, 145, 108, 151, 103, 146, 123, 111]),
    heartRate: series([79, 75, 76, 88, 128, 96, 132, 91, 143, 88, 80]),
    events: [
      { hour: 7.75, kind: 'meal', value: 116 },
      { hour: 12.5, kind: 'meal', value: 130 },
      { hour: 19.17, kind: 'meal', value: 116 },
      { hour: 8, kind: 'medication', value: 66 },
      { hour: 20, kind: 'medication', value: 66 },
    ],
    meals: [
      { name: 'Breakfast', summary: 'Oatmeal with fruit', time: '7:45 AM', carbs: 48, peak: 142 },
      { name: 'Lunch', summary: 'Chicken rice bowl', time: '12:30 PM', carbs: 62, peak: 155 },
      { name: 'Dinner', summary: 'Pasta and salad', time: '7:10 PM', carbs: 70, peak: 148 },
    ],
    mealResponses: mealResponses([0, 18, 36, 26, 8], [0, 22, 43, 31, 10], [0, 20, 39, 28, 9]),
  },
};
