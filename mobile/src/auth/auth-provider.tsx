import { useAuth as useClerkAuth, useClerk, useUser } from '@clerk/expo';
import {
  createContext,
  type PropsWithChildren,
  useCallback,
  useContext,
  useMemo,
} from 'react';

import { demoPersonIdFromMetadata, type AuthenticatedUser, roleFromMetadata } from './types';

type AuthContextValue = {
  isReady: boolean;
  user: AuthenticatedUser | null;
  getSessionToken: () => Promise<string | null>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const { getToken, isLoaded: isAuthLoaded, isSignedIn } = useClerkAuth();
  const { isLoaded: isUserLoaded, user: clerkUser } = useUser();
  const { signOut: clerkSignOut } = useClerk();

  const resolvedUser = useMemo<AuthenticatedUser | null>(() => {
    if (!isSignedIn || !clerkUser) return null;

    const email = clerkUser.primaryEmailAddress?.emailAddress ?? '';
    return {
      email,
      name: clerkUser.fullName ?? clerkUser.firstName ?? email ?? 'Nutrix member',
      personId: demoPersonIdFromMetadata(clerkUser.publicMetadata),
      role: roleFromMetadata(clerkUser.publicMetadata),
    };
  }, [clerkUser, isSignedIn]);

  const signOut = useCallback(async () => {
    await clerkSignOut();
  }, [clerkSignOut]);

  const getSessionToken = useCallback(async () => {
    if (!isSignedIn) return null;
    return getToken();
  }, [getToken, isSignedIn]);

  const value = useMemo<AuthContextValue>(
    () => ({
      isReady: isAuthLoaded && isUserLoaded,
      user: resolvedUser,
      getSessionToken,
      signOut,
    }),
    [
      getSessionToken,
      isAuthLoaded,
      isUserLoaded,
      resolvedUser,
      signOut,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside AuthProvider.');
  return context;
}
