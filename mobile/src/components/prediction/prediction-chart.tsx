import { Text, useWindowDimensions, View } from 'react-native';
import Svg, { Circle, Line, Path, Text as SvgText } from 'react-native-svg';

import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

export type GlucoseChartData = {
  lowerBound: readonly number[];
  minutes: readonly number[];
  observed: readonly number[];
  predicted: readonly number[];
  upperBound: readonly number[];
};

type PredictionChartProps = {
  data: GlucoseChartData;
  day: number;
  peakGlucose: number;
  timeToPeak: number;
};

const chartHeight = 210;
const chartPadding = { bottom: 28, left: 34, right: 8, top: 12 };
const minGlucose = 80;
const maxGlucose = 180;
const glucoseTickStep = 20;

function buildLinePath(values: readonly number[], toX: (index: number) => number, toY: (value: number) => number) {
  return values.map((value, index) => `${index === 0 ? 'M' : 'L'} ${toX(index)} ${toY(value)}`).join(' ');
}

function ChartLegend() {
  const { colors, text } = usePatientTheme();
  return (
    <View className="mt-3 flex-row flex-wrap gap-x-5 gap-y-2">
      <View className="flex-row items-center">
        <View className="mr-2 h-1 w-5 rounded-full" style={{ backgroundColor: colors.medication }} />
        <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>Predicted</Text>
      </View>
      <View className="flex-row items-center">
        <View className="mr-2 h-1 w-5 rounded-full" style={{ backgroundColor: colors.success }} />
        <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>Observed</Text>
      </View>
      <View className="flex-row items-center">
        <View className="mr-2 h-3 w-4 rounded" style={{ backgroundColor: colors.accent }} />
        <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>Uncertainty</Text>
      </View>
      <View className="flex-row items-center">
        <View className="mr-2 h-3 w-3 rounded-full border-2 bg-surface" style={{ borderColor: colors.medication }} />
        <Text className="text-sm text-copy" style={[text.label, { color: colors.mutedForeground }]}>Model peak</Text>
      </View>
    </View>
  );
}

export function PredictionChart({ data, day, peakGlucose, timeToPeak }: PredictionChartProps) {
  const { cardStyle, colors, isSeniorMode, text } = usePatientTheme();
  const { width } = useWindowDimensions();
  const xLabelIndexes = [0, 0.25, 0.5, 0.75, 1].map((fraction) =>
    Math.round((data.minutes.length - 1) * fraction),
  );
  const chartWidth = Math.max(260, width - 88);
  const plotWidth = chartWidth - chartPadding.left - chartPadding.right;
  const plotHeight = chartHeight - chartPadding.top - chartPadding.bottom;
  const chartMax = Math.max(
    maxGlucose,
    Math.ceil(Math.max(peakGlucose, ...data.observed, ...data.predicted, ...data.upperBound) / glucoseTickStep) * glucoseTickStep,
  );
  const yTicks = Array.from({ length: (chartMax - minGlucose) / glucoseTickStep + 1 }, (_, index) => minGlucose + index * glucoseTickStep);
  const timeRange = data.minutes[data.minutes.length - 1] - data.minutes[0];
  const toX = (index: number) => chartPadding.left + (index / (data.minutes.length - 1)) * plotWidth;
  const peakX = chartPadding.left + ((timeToPeak - data.minutes[0]) / timeRange) * plotWidth;
  const toY = (value: number) =>
    chartPadding.top + ((chartMax - value) / (chartMax - minGlucose)) * plotHeight;
  const uncertaintyPath = [
    ...data.upperBound.map((value, index) => `${index === 0 ? 'M' : 'L'} ${toX(index)} ${toY(value)}`),
    ...data.lowerBound
      .map((value, index) => ({ index: data.lowerBound.length - 1 - index, value }))
      .map(({ index, value }) => `L ${toX(index)} ${toY(value)}`),
    'Z',
  ].join(' ');

  return (
    <View className="app-card-lg" style={cardStyle}>
      <Text className="text-xl font-semibold text-ink" style={[text.cardTitle, { color: colors.foreground }]}>Predicted vs observed · Day {day}</Text>
      <ChartLegend />
      <View className="mt-2 items-center">
        <Svg height={chartHeight} width={chartWidth} accessibilityLabel={`Predicted and observed glucose chart for Day ${day}`}>
          <Path d={uncertaintyPath} fill={colors.accent} opacity={0.55} />
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
              key={tick}
              textAnchor="end"
              x={chartPadding.left - 6}
              y={toY(tick) + 4}>
              {tick}
            </SvgText>
          ))}
          <Path d={buildLinePath(data.observed, toX, toY)} fill="none" stroke={colors.success} strokeWidth={3} />
          <Path
            d={buildLinePath(data.predicted, toX, toY)}
            fill="none"
            stroke={colors.medication}
            strokeDasharray="7 5"
            strokeLinecap="round"
            strokeWidth={3}
          />
          <Circle cx={peakX} cy={toY(peakGlucose)} fill={colors.card} r={5} stroke={colors.medication} strokeWidth={3} />
          <SvgText
            fill={colors.medication}
            fontSize={isSeniorMode ? 13 : 11}
            fontWeight="600"
            textAnchor="middle"
            x={peakX}
            y={Math.max(chartPadding.top + 10, toY(peakGlucose) - 8)}>
            Peak {Math.round(peakGlucose)}
          </SvgText>
          {xLabelIndexes.map((index) => (
            <SvgText
              fill={colors.mutedForeground}
              fontSize={isSeniorMode ? 13 : 11}
              key={data.minutes[index]}
              textAnchor="middle"
              x={toX(index)}
              y={chartHeight - 7}>
              {data.minutes[index]} min
            </SvgText>
          ))}
          {xLabelIndexes.map((index) => (
            <Circle
              cx={toX(index)}
              cy={toY(data.predicted[index])}
              fill={colors.medication}
              key={data.minutes[index]}
              r={index === 0 ? 4.5 : 4}
            />
          ))}
        </Svg>
      </View>
      <Text className="mt-2 text-xs leading-4 text-copy" style={[text.caption, { color: colors.mutedForeground }]}>
        The dashed line is a 5-minute average trajectory. The model predicts the peak independently, so it may sit above the curve.
      </Text>
    </View>
  );
}
