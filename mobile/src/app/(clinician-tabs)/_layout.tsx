import { Tabs } from 'expo-router';
import type { BottomTabBarProps } from 'expo-router/build/react-navigation/bottom-tabs';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useReducedMotion } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { icons } from '../../../constants/icons';
import { colors, components } from '../../../constants/theme';
import { useLanguage } from '@/features/localization/language-context';

const clinicianTabs = [
  { icon: icons.home, name: 'index', translationKey: 'clinicianHome' },
  { icon: icons.settings, name: 'settings', translationKey: 'settings' },
] as const;
const tabBar = components.tabBar;

function ClinicianTabBar({ descriptors, insets, navigation, state }: BottomTabBarProps) {
  const { t } = useLanguage();

  return (
    <View style={[styles.tabBar, { height: tabBar.height + insets.bottom, paddingBottom: insets.bottom }]}>
      {state.routes.map((route, index) => {
        const tab = clinicianTabs.find((item) => item.name === route.name);
        if (!tab) return null;
        const isFocused = index === state.index;
        const label = t(tab.translationKey);

        return (
          <Pressable
            accessibilityLabel={descriptors[route.key].options.tabBarAccessibilityLabel ?? label}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            key={route.key}
            onLongPress={() => navigation.emit({ type: 'tabLongPress', target: route.key })}
            onPress={() => {
              const event = navigation.emit({ type: 'tabPress', target: route.key, canPreventDefault: true });
              if (!isFocused && !event.defaultPrevented) navigation.navigate(route.name);
            }}
            style={styles.tabButton}>
            <Image source={tab.icon} resizeMode="contain" style={[styles.tabIcon, { tintColor: isFocused ? tabBar.activeColor : tabBar.inactiveColor }]} />
            <Text style={[styles.tabLabel, { color: isFocused ? tabBar.activeColor : tabBar.inactiveColor }]}>{label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

export default function ClinicianTabsLayout() {
  const insets = useSafeAreaInsets();
  const reduceMotion = useReducedMotion();

  return (
    <Tabs
      screenOptions={{ animation: reduceMotion ? 'none' : 'fade', headerShown: false, sceneStyle: { paddingBottom: tabBar.height + insets.bottom }, tabBarShowLabel: false }}
      tabBar={(props) => <ClinicianTabBar {...props} />}>
      {clinicianTabs.map((tab) => <Tabs.Screen key={tab.name} name={tab.name} options={{ title: tab.translationKey }} />)}
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: { alignItems: 'center', backgroundColor: colors.card, borderTopColor: colors.border, borderTopWidth: StyleSheet.hairlineWidth, bottom: 0, elevation: 8, flexDirection: 'row', left: 0, position: 'absolute', right: 0, shadowColor: colors.foreground, shadowOffset: { width: 0, height: -2 }, shadowOpacity: 0.06, shadowRadius: 8 },
  tabButton: { alignItems: 'center', flex: 1, height: '100%', justifyContent: 'center', paddingTop: 6 },
  tabIcon: { height: tabBar.iconSize, width: tabBar.iconSize },
  tabLabel: { fontSize: 12, fontWeight: '600', marginTop: tabBar.labelTopMargin },
});
