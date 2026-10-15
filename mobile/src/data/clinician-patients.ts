import { getDemoHistory } from '@/data/demo-history';
import type { DemoPersonId } from '@/auth/types';

export type ClinicianPatient = {
  id: DemoPersonId;
  initials: string;
  name: string;
  status: 'lowRangeReview' | 'highRangeReview';
  timeInRange: number;
  demographics: {
    age: number;
    bmi: number;
    diabetesType: 'Type 1' | 'Type 2';
    diagnosedAge: number;
    ethnicity: string;
    gender: 'Female' | 'Male' | 'Non-binary';
    hba1c: number;
    height: string;
    isPregnant?: 'Yes' | 'No' | 'Not sure';
    weight: string;
  };
};

const patientIds: readonly DemoPersonId[] = ['cgmacros-002', 'cgmacros-003'];

const patientProfiles: Record<DemoPersonId, Omit<ClinicianPatient, 'id' | 'timeInRange'>> = {
  'cgmacros-002': {
    initials: 'JD',
    name: 'Jordan Davis',
    status: 'lowRangeReview',
    demographics: {
      age: 42,
      bmi: 25,
      diabetesType: 'Type 2',
      diagnosedAge: 35,
      ethnicity: 'Asian',
      gender: 'Female',
      hba1c: 7.2,
      height: "5' 5\"",
      isPregnant: 'No',
      weight: '150 lb',
    },
  },
  'cgmacros-003': {
    initials: 'MC',
    name: 'Maya Chen',
    status: 'highRangeReview',
    demographics: {
      age: 29,
      bmi: 22.7,
      diabetesType: 'Type 1',
      diagnosedAge: 12,
      ethnicity: 'Hispanic or Latino',
      gender: 'Female',
      hba1c: 7.8,
      height: "5' 4\"",
      isPregnant: 'No',
      weight: '132 lb',
    },
  },
};

export const clinicianPatients: readonly ClinicianPatient[] = patientIds.map((id) => ({
  id,
  ...patientProfiles[id],
  timeInRange: getDemoHistory(id).timeInRange.inRange,
}));

export function getClinicianPatient(patientId: string) {
  return clinicianPatients.find((patient) => patient.id === patientId);
}
