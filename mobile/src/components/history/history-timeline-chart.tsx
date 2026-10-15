import { Text, useWindowDimensions, View } from 'react-native';
import Svg, { Line, Path, Polygon, Text as SvgText } from 'react-native-svg';

import type { HistoryDay, HistorySeriesPoint } from '@/data/history';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

type HistoryTimelineChartProps = {
  day: HistoryDay;
};

const chartHeight = 180;
const chartPadding = { bottom: 28, left: 30, right: 8, top: 10 };
const xTicks = [0, 6, 12, 18, 24] as const;

function makePath(points: readonly HistorySeriesPoint[], toX: (hour: number) => number, toY: (value: number) => number) {
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${toX(point.hour)} ${toY(point.value)}`).join(' ');
}

function LegendItem({ color, diamond = false, label, textStyle, textColor }: { color: string; diamond?: boolean; label: string; textColor: string; textStyle: { fontSize?: number; lineHeight?: number } }) {
  return (
    <View className="flex-row items-center">
      <View
        className={diamond ? 'mr-1.5 h-2.5 w-2.5 rotate-45' : 'mr-1.5 h-3 w-3 rounded-full'}
        style={{ backgroundColor: color }}
      />
      <Text className="text-copy" style={[textStyle, { color: textColor }]}>{label}</Text>
    </View>
  );
}

export function HistoryTimelineChart({ day }: HistoryTimelineChartProps) {
  const { t } = useLanguage();
  const { cardStyle, colors, isSeniorMode, text } = usePatientTheme();
  const { width } = useWindowDimensions();
  const chartWidth = Math.max(220, width - 80);
  const plotWidth = chartWidth - chartPadding.left - chartPadding.right;
  const plotHeight = chartHeight - chartPadding.top - chartPadding.bottom;
  const plottedValues = [...day.glucose, ...day.heartRate, ...day.events].map((point) => point.value);
  const minimum = Math.max(0, Math.floor((Math.min(...plottedValues) - 10) / 20) * 20);
  const maximum = Math.ceil((Math.max(...plottedValues) + 10) / 20) * 20;
  const yTicks = Array.from({ length: 5 }, (_, index) => minimum + ((maximum - minimum) / 4) * index);
  const toX = (hour: number) => chartPadding.left + (hour / 24) * plotWidth;
  const toY = (value: number) => chartPadding.top + ((maximum - value) / Math.max(1, maximum - minimum)) * plotHeight;

  return (
    <View className="app-card mt-4" style={cardStyle}>
      <Text className="app-card-title" style={[text.cardTitle, { color: colors.foreground }]}>
        {t('glucoseHeartRate', { day: day.id })}
      </Text>
      <View className="mt-2 items-center">
        <Svg height={chartHeight} width={chartWidth} accessibilityLabel={`24-hour history chart for ${day.id}`}>
          {yTicks.map((tick) => (
            <Line
              key={tick}
              stroke={colors.border}
              strokeWidth={1}
              x1={chartPadding.left}
              x2={chartWidth - chartPadding.right}
              y1={toY(tick)}
              y2={toY(tick)}
            />
          ))}
          {yTicks.map((tick) => (
            <SvgText
              fill={colors.mutedForeground}
              fontSize={isSeniorMode ? 13 : 11}
              key={`label-${tick}`}
              textAnchor="end"
              x={chartPadding.left - 5}
              y={toY(tick) + 4}>
              {tick}
            </SvgText>
          ))}
          <Path d={makePath(day.heartRate, toX, toY)} fill="none" stroke={colors.heartRate} strokeWidth={2} />
          <Path d={makePath(day.glucose, toX, toY)} fill="none" stroke={colors.success} strokeWidth={3} />
          {day.events.map((event, index) => {
            const x = toX(event.hour);
            const y = toY(event.value);

            if (event.kind === 'meal') {
              return null;
            }

            return (
              <Polygon
                fill={colors.medication}
                key={`${event.kind}-${index}`}
                points={`${x},${y - 6} ${x + 6},${y} ${x},${y + 6} ${x - 6},${y}`}
              />
            );
          })}
          {xTicks.map((tick) => (
            <SvgText
              fill={colors.mutedForeground}
              fontSize={isSeniorMode ? 13 : 11}
              key={`time-${tick}`}
              textAnchor={tick === 0 ? 'start' : tick === 24 ? 'end' : 'middle'}
              x={toX(tick)}
              y={chartHeight - 6}>
              {tick}h
            </SvgText>
          ))}
        </Svg>
      </View>
      <Text className="mt-2 text-xs text-copy" style={[text.caption, { color: colors.mutedForeground }]}>{t('hoursSince')}</Text>
      <View className="mt-2 flex-row flex-wrap gap-x-4 gap-y-2">
        <LegendItem color={colors.success} label={t('glucose')} textColor={colors.mutedForeground} textStyle={text.caption} />
        <LegendItem color={colors.heartRate} label={t('heartRate')} textColor={colors.mutedForeground} textStyle={text.caption} />
      </View>
    </View>
  );
}
