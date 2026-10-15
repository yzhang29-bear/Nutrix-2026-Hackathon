import { Redirect } from 'expo-router';
import { ActivityIndicator, View } from 'react-native';

import { useAuth } from '@/auth/auth-provider';
import { useHealthQuestionnaire } from '@/features/profile/health-questionnaire-context';

export default function IndexScreen() {
  const { isReady, user } = useAuth();
  const { isQuestionnairePending } = useHealthQuestionnaire();

  if (!isReady) {
    return <View className="app-screen items-center justify-center"><ActivityIndicator /></View>;
  }

  if (!user) return <Redirect href="/(auth)/sign-in" />;
  if (user.role === 'Clinician') return <Redirect href="/(clinician-tabs)" />;

  return <Redirect href={isQuestionnairePending ? '/health-questionnaire' : '/(tabs)'} />;
}
