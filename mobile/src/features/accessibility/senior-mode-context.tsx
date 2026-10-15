import { createContext, useContext, useEffect, useMemo, useState, type PropsWithChildren } from 'react';
import type { TextStyle, ViewStyle } from 'react-native';

import { useAuth } from '@/auth/auth-provider';
import { colors as defaultColors } from '../../../constants/theme';

type PatientColors = Record<keyof typeof defaultColors, string> & {
  statusHigh: string;
  statusLow: string;
  statusNormal: string;
};

const standardColors: PatientColors = {
  ...defaultColors,
  statusHigh: '#f8d2dc',
  statusLow: '#ffe6bd',
  statusNormal: '#d6efe6',
};

const seniorColors: PatientColors = {
  background: '#ffffff',
  foreground: '#111827',
  card: '#ffffff',
  muted: '#d1d5db',
  mutedForeground: '#4b5563',
  primary: '#0F3FD1',
  primaryHover: '#0b2e9b',
  primarySoft: '#dbe7ff',
  onPrimary: '#ffffff',
  onPrimarySoft: '#0F3FD1',
  secondary: '#D1FAE5',
  onSecondary: '#065f46',
  accent: '#9A12F3',
  heartRate: '#EF4444',
  meal: '#0F3FD1',
  medication: '#9A12F3',
  border: '#9ca3af',
  borderPrimary: '#0F3FD1',
  success: '#047857',
  destructive: '#EF4444',
  statusHigh: '#fee2e2',
  statusLow: '#f3e8ff',
  statusNormal: '#D1FAE5',
};

type PatientTheme = {
  cardStyle: ViewStyle;
  colors: PatientColors;
  controlStyle: ViewStyle;
  primaryButtonStyle: ViewStyle;
  isSeniorMode: boolean;
  setSeniorMode: (enabled: boolean) => void;
  text: {
    body: TextStyle;
    button: TextStyle;
    caption: TextStyle;
    cardTitle: TextStyle;
    label: TextStyle;
    pageTitle: TextStyle;
    reading: TextStyle;
  };
};

const PatientThemeContext = createContext<PatientTheme | null>(null);

export function PatientThemeProvider({ children }: PropsWithChildren) {
  const { user } = useAuth();
  const [enabled, setEnabled] = useState(false);
  const canUseSeniorMode = Boolean(user && user.role !== 'Clinician');
  const isSeniorMode = canUseSeniorMode && enabled;

  useEffect(() => {
    if (!user || user.role === 'Clinician') setEnabled(false);
  }, [user]);

  const value = useMemo<PatientTheme>(() => {
    const colors = isSeniorMode ? seniorColors : standardColors;
    const scale = 1.16;
    const seniorText = {
      caption: { fontSize: Math.round(12 * scale), lineHeight: Math.round(17 * scale) },
      label: { fontSize: Math.round(14 * scale), lineHeight: Math.round(20 * scale) },
      body: { fontSize: Math.round(16 * scale), lineHeight: Math.round(24 * scale) },
      button: { fontSize: Math.round(16 * scale) },
      cardTitle: { fontSize: Math.round(18 * scale), lineHeight: Math.round(24 * scale) },
      pageTitle: { fontSize: Math.round(36 * scale), lineHeight: Math.round(43 * scale) },
      reading: { fontSize: Math.round(30 * scale), lineHeight: Math.round(36 * scale) },
    };

    return {
      colors,
      isSeniorMode,
      setSeniorMode: setEnabled,
      cardStyle: isSeniorMode ? { backgroundColor: colors.card, borderColor: colors.border } : {},
      controlStyle: isSeniorMode ? { minHeight: 52 } : {},
      primaryButtonStyle: isSeniorMode ? { backgroundColor: colors.primary, minHeight: 68 } : {},
      text: isSeniorMode ? seniorText : { caption: {}, label: {}, body: {}, button: {}, cardTitle: {}, pageTitle: {}, reading: {} },
    };
  }, [isSeniorMode]);

  return <PatientThemeContext.Provider value={value}>{children}</PatientThemeContext.Provider>;
}

export function usePatientTheme() {
  const value = useContext(PatientThemeContext);
  if (!value) throw new Error('usePatientTheme must be used inside PatientThemeProvider.');
  return value;
}
