
import globals from 'globals';
import js from '@eslint/js';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';

export default [
  // Ignore specific files/directories
  {
    ignores: ['dist'],
  },
  // Basic JavaScript rules
  js.configs.recommended,
  // React plugin rules
  {
    files: ['**/*.{js,jsx}'], // Target JavaScript and JSX files
    plugins: {
      react,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    languageOptions: {
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: globals.browser,
    },
    rules: {
      // Recommended React rules
      ...react.configs.recommended.rules,
      // Recommended React Hooks rules
      ...reactHooks.configs.recommended.rules,
      // React Refresh rules for Vite
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // Add any other specific rules you need for JSX
      // 'react/prop-types': 'off', // Often turned off in modern React
      // 'react/react-in-jsx-scope': 'off', // Not needed with new JSX transform
    },
    settings: {
      react: {
        version: 'detect', // Auto-detect React version
      },
    },
  },
];
