import { createContext, useContext, useMemo, useState, type PropsWithChildren } from 'react';

export const genders = ['Female', 'Male', 'Non-binary', 'Prefer not to say'] as const;
export const ethnicities = [
  'Asian',
  'Black or African American',
  'Hispanic or Latino',
  'White',
  'American Indian or Alaska Native',
  'Native Hawaiian or Pacific Islander',
  'Middle Eastern or North African',
  'Other',
  'Prefer not to say',
] as const;
export const diabetesTypes = ['Type 1', 'Type 2', 'Gestational', 'Other', 'Prefer not to say'] as const;
export const managementOptions = ['Basal insulin', 'Bolus insulin', 'Other insulin', 'Other'] as const;

export type Gender = (typeof genders)[number];
export type Ethnicity = (typeof ethnicities)[number];
export type DiabetesType = (typeof diabetesTypes)[number];
export type ManagementOption = (typeof managementOptions)[number];
export type PregnancyStatus = 'Yes' | 'No' | 'Not sure';

export type HealthQuestionnaire = {
  age: string;
  diabetesManagement: ManagementOption[];
  diabetesType?: DiabetesType;
  ethnicity: Ethnicity[];
  gender?: Gender;
  height: string;
  isPregnant?: PregnancyStatus;
  weight: string;
};

type HealthQuestionnaireContextValue = {
  answers: HealthQuestionnaire;
  beginQuestionnaire: () => void;
  completeQuestionnaire: () => void;
  isQuestionnairePending: boolean;
  resetAnswers: () => void;
  updateAnswers: (changes: Partial<HealthQuestionnaire>) => void;
};

const initialAnswers: HealthQuestionnaire = {
  age: '',
  diabetesManagement: [],
  ethnicity: [],
  height: '',
  weight: '',
};

const HealthQuestionnaireContext = createContext<HealthQuestionnaireContextValue | null>(null);

export function HealthQuestionnaireProvider({ children }: PropsWithChildren) {
  const [answers, setAnswers] = useState<HealthQuestionnaire>(initialAnswers);
  const [isQuestionnairePending, setIsQuestionnairePending] = useState(false);

  const value = useMemo<HealthQuestionnaireContextValue>(
    () => ({
      answers,
      isQuestionnairePending,
      beginQuestionnaire() {
        setIsQuestionnairePending(true);
      },
      completeQuestionnaire() {
        setIsQuestionnairePending(false);
      },
      updateAnswers(changes) {
        setAnswers((current) => ({ ...current, ...changes }));
      },
      resetAnswers() {
        setAnswers(initialAnswers);
      },
    }),
    [answers, isQuestionnairePending],
  );

  return <HealthQuestionnaireContext.Provider value={value}>{children}</HealthQuestionnaireContext.Provider>;
}

export function useHealthQuestionnaire() {
  const value = useContext(HealthQuestionnaireContext);

  if (!value) {
    throw new Error('useHealthQuestionnaire must be used inside HealthQuestionnaireProvider.');
  }

  return value;
}
