const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    rules: {
      // Role-tab mismatch handling deliberately signs out after Clerk restores
      // the current session and loads the trusted staff metadata.
      'react-hooks/set-state-in-effect': 'off',
    },
  },
]);
