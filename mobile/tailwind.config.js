/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        canvas: '#f6faf9',
        surface: '#ffffff',
        ink: '#35505a',
        copy: '#9aa8a6',
        line: '#e6ebea',
        brand: {
          DEFAULT: '#8fd3be',
          hover: '#52b69a',
          soft: '#d6efe6',
          subtle: '#eef8f4',
          dark: '#2e6b58',
        },
        'on-brand': '#1f4d40',
        secondary: '#ffd9a0',
        'secondary-hover': '#ffa92c',
        'on-secondary': '#6b4a12',
        accent: '#a9c4e2',
        glucose: '#76c893',
        heart: '#f2b8c6',
        meal: '#ffa92c',
        medication: '#6f8fb3',
        prediction: '#6f8fb3',
        warning: '#ffa92c',
        danger: '#7a2e45',
        'status-low': '#ffe6bd',
        'status-normal': '#d6efe6',
        'status-high': '#f8d2dc',
      },
      borderRadius: {
        card: '16px',
        'card-lg': '16px',
      },
      boxShadow: {
        'token-sm': '0 2px 4px rgba(12, 12, 32, 0.10)',
        'token-md': '0 4px 8px rgba(12, 12, 32, 0.10)',
        'token-lg': '0 4px 8px rgba(12, 12, 32, 0.15)',
        glass: '0 4px 16px rgba(12, 12, 32, 0.06)',
        card: '0 4px 16px rgba(12, 12, 32, 0.06)',
      },
      fontFamily: {
        sans: ['Atkinson Hyperlegible Next', 'system-ui', 'sans-serif'],
        numeric: ['Atkinson Hyperlegible Mono', 'Atkinson Hyperlegible Next', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
