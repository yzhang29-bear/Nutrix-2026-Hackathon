const app = require('./app.json').expo;

module.exports = {
  ...app,
  plugins: [...(app.plugins ?? []), 'expo-secure-store', '@clerk/expo', '@react-native-community/datetimepicker'],
};
