import insights from "@/assets/icons/insights.png";
import home from "@/assets/icons/home.png";
import meal from "@/assets/icons/meal.png";
import settings from "@/assets/icons/settings.png";

export const icons = {
  home,
  meal,
  settings,
  insights,
} as const;

export type IconKey = keyof typeof icons;