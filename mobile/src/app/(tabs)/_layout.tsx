import { Tabs } from 'expo-router';
import type { BottomTabBarProps } from 'expo-router/build/react-navigation/bottom-tabs';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useReducedMotion } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { tabs } from '../../../constants/data';
import { components } from '../../../constants/theme';
import { useLanguage } from '@/features/localization/language-context';
import { usePatientTheme } from '@/features/accessibility/senior-mode-context';

const tabBar = components.tabBar;
function PatientTabBar({ state, descriptors, navigation, insets }: BottomTabBarProps) {
  const { t } = useLanguage();
  const { colors, isSeniorMode, text } = usePatientTheme();
  const barHeight = isSeniorMode ? 92 : tabBar.height;
  return (
    <View
      style={[
        styles.tabBar,
        {
          height: barHeight + insets.bottom,
          paddingBottom: insets.bottom,
          backgroundColor: colors.card,
          borderTopColor: colors.border,
          shadowColor: colors.foreground,
        },
      ]}>
      {state.routes.map((route, index) => {
        const tab = tabs.find((item) => item.name === route.name);

        if (!tab) {
          return null;
        }

        const isFocused = state.index === index;
        const options = descriptors[route.key].options;
        const translatedTitle = t(tab.name === 'index' ? 'today' : tab.name);
        const accessibilityLabel = options.tabBarAccessibilityLabel ?? translatedTitle;

        function onPress() {
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });

          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name);
          }
        }

        return (
          <Pressable
            key={route.key}
            accessibilityLabel={accessibilityLabel}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            onLongPress={() => navigation.emit({ type: 'tabLongPress', target: route.key })}
            onPress={onPress}
            style={[styles.tabButton, isSeniorMode ? { minHeight: barHeight } : undefined]}>
            <Image
              source={tab.icon}
              resizeMode="contain"
              style={[styles.tabIcon, isSeniorMode ? { height: 32, tintColor: isFocused ? colors.primary : colors.mutedForeground, width: 32 } : { tintColor: isFocused ? colors.onPrimarySoft : colors.mutedForeground }]}
            />
            <Text style={[styles.tabLabel, isSeniorMode ? text.caption : undefined, { color: isFocused ? colors.primary : colors.mutedForeground }]}>
              {translatedTitle}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

export default function PatientTabsLayout() {
  const reduceMotion = useReducedMotion();
  const insets = useSafeAreaInsets();
  const { isSeniorMode } = usePatientTheme();
  const barHeight = isSeniorMode ? 92 : tabBar.height;

  return (
    <Tabs
      tabBar={(props) => <PatientTabBar {...props} />}
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        animation: reduceMotion ? 'none' : 'fade',
        sceneStyle: { paddingBottom: barHeight + insets.bottom },
      }}>
      {tabs.map((tab) => (
        <Tabs.Screen
          key={tab.name}
          name={tab.name}
          options={{
            title: tab.title,
          }}
        />
      ))}
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    borderTopWidth: StyleSheet.hairlineWidth,
    elevation: 8,
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
  },
  tabButton: {
    flex: 1,
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 6,
  },
  tabIcon: {
    width: tabBar.iconSize,
    height: tabBar.iconSize,
  },
  tabLabel: {
    marginTop: tabBar.labelTopMargin,
    fontSize: 12,
    fontWeight: '600',
  },
});
