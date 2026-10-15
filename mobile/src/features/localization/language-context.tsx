import { createContext, useContext, useMemo, useState, type PropsWithChildren } from 'react';

export const languages = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Español' },
  { code: 'zh-Hans', label: '简体中文' },
] as const;

export type LanguageCode = (typeof languages)[number]['code'];
const translations: Record<LanguageCode, Record<string, string>> = {
  en: {
    accountFallback: 'Your account',
    cancel: 'Cancel',
    today: 'Today', history: 'History', insights: 'Insights', settings: 'Settings', clinicianHome: 'Home', clinicianDashboard: 'Dashboard', patients: 'Patients', patientCount: '{count} patients', searchPatients: 'Search patients', timeInRangeShort: 'TIR {percent}%', syncedToday: 'synced today', lowRangeReview: 'Review low range', highRangeReview: 'Review high range', noPatientsFound: 'No patients found', openPatientProfile: 'Open patient profile', patientProfile: 'Patient profile', yearsOld: '{age} y', bmi: 'BMI', pregnant: 'Pregnant', hba1c: 'HbA1c', sevenDaySummary: '7-day summary report', summaryReportDescription: 'Recorded glucose data from the last seven days.', averageGlucose: 'Average glucose', recordedMeals: 'Recorded meals', latestDayAverage: 'Latest day average', 'Type 1': 'Type 1', 'Type 2': 'Type 2', 'Asian': 'Asian', 'Hispanic or Latino': 'Hispanic or Latino',
    todayDescription: 'Your meal and glucose forecast will appear here.',
    personalizationStatus: 'Personalization status', modelReady: 'Your model is ready for meal predictions.',
    daysCollected: '{days} of {required} days collected. Predictions unlock after your baseline is ready.',
    checkingStatus: 'Checking your personalization status…', mealPhoto: 'Meal photo',
    addPhotoDescription: 'Add a photo to help capture your meal details.', addMealPhoto: 'Add meal photo', preparingPhoto: 'Preparing your photo…',
    healthQuestionnaire: 'Health questionnaire', skip: 'Skip', back: 'Back', continue: 'Continue', finish: 'Finish',
    aboutYou: 'About you', diabetesHistory: 'Diabetes history', tellUs: 'Tell us about your health',
    questionnaireDescription: 'About 5 minutes. Your answers personalize glucose predictions during this session.',
    diabetesDescription: 'This helps personalize your glucose insights and predictions.', age: 'Age', height: 'Height', weight: 'Weight', gender: 'Gender',
    pregnancy: 'Are you currently pregnant?', pregnancyDescription: 'Pregnancy changes glucose targets. Your care team will adjust guidance.', ethnicity: 'Ethnicity', selectAll: 'Select all that apply',
    diabetesType: 'Diabetes type', diabetesManagement: 'How do you manage diabetes?', notProvided: 'Not provided', management: 'Management', sessionOnly: 'SESSION ONLY',
    editHealthProfile: 'Edit health profile', saveChanges: 'Save changes', backToSettings: 'Back to settings',
    addMeal: 'Add meal', mealType: 'Meal type', mealName: 'Meal name', carbohydrates: 'Carbohydrates', mealTime: 'Meal time', photo: 'Photo',
    nutritionEstimation: 'Used for nutrition estimation', replacePhoto: 'Replace photo', optionalPrediction: 'Optional for this demo prediction.', predictResponse: 'Predict glucose response',
    photoCancelled: 'Photo selection cancelled.', photoPermissionDenied: 'Camera access was denied. Enable it in Settings to take a meal photo.', photoNormalizationError: 'We could not prepare that photo. Try another image.', photoOversize: 'That photo is over 8 MB after preparation. Try another image.', currentTimePreview: 'Uses the current time on web preview.', discardMealDraft: 'Discard meal draft and return home', mealNamePlaceholder: 'e.g. Oatmeal with fruit',
    predictionPersonalization: 'Uses your last 7 days of CGM, heart rate, and medication data to personalize the prediction.',
    enterMealName: 'Enter a meal name before creating a prediction.', predictionUnavailable: 'Unable to create a prediction. Try again.',
    historyFootnote: 'Each day uses recorded CGMacros samples; meal details are available for the final sampled day.',
    glucoseHeartRate: 'Your glucose and heart rate history for {day}', hoursSince: 'Hours since the sampled day began.', glucose: 'Glucose', heartRate: 'Heart rate',
    dailyAverage: 'Daily average glucose (mg/dL)', loggedMeals: 'Logged meals · Day {day}', carbs: 'g carbs', peak: 'peak mg/dL',
    timeInRange: 'Time in range · last 7 days', low: 'Low <70', inRange: 'In range', high: 'High >140',
    observedMealResponse: 'Observed meal response · first 7 days', meanGlucoseChange: 'Mean observed glucose change', meanRise: 'Mean observed rise at 60 min: +{value} mg/dL across {count} complete meal-response windows.',
    recordedCoverage: 'Recorded data coverage', coverageText: '{days} days of CGM and {meals} recorded meals in the displayed baseline.', responseText: '{count} meals had complete 0–120 minute glucose windows for the response calculation.',
    mealPrediction: 'Meal prediction', predictedObserved: 'Predicted · observed', glucoseAfterMeal: 'Glucose after meal', confidence: 'Confidence', goHome: 'Go back to home page',
    peakLabel: 'Peak', timeToPeak: 'Time to peak', predicted: 'Predicted', observed: 'Observed', uncertainty: 'Uncertainty', modelPeak: 'Model peak', topDrivers: 'Top drivers for this prediction',
    Breakfast: 'Breakfast', Lunch: 'Lunch', Dinner: 'Dinner', Snack: 'Snack', Female: 'Female', Male: 'Male', 'Non-binary': 'Non-binary', 'Prefer not to say': 'Prefer not to say', Yes: 'Yes', No: 'No', 'Not sure': 'Not sure',
    healthProfile: 'Health profile',
    healthProfileDescription: 'Review and edit your questionnaire answers',
    language: 'Language',
    languageDescription: 'Choose the language for Nutrix',
    preferences: 'Your preferences',
    preferencesDescription: 'Dietary preferences, glucose data, notifications, and privacy controls will live here.',
    signOut: 'Sign out',
    signOutBody: 'You will need your email and password to access Nutrix again.',
    signOutTitle: 'Sign out?',
    seniorMode: 'Senior mode', seniorModeDescription: 'Use higher-contrast colors, larger text, and larger controls.',
  },
  es: {
    accountFallback: 'Tu cuenta',
    cancel: 'Cancelar',
    healthProfile: 'Perfil de salud',
    healthProfileDescription: 'Revisa y edita las respuestas de tu cuestionario',
    language: 'Idioma',
    languageDescription: 'Elige el idioma de Nutrix',
    preferences: 'Tus preferencias',
    preferencesDescription: 'Aquí estarán tus preferencias alimentarias, datos de glucosa, notificaciones y controles de privacidad.',
    today: 'Hoy', history: 'Historial', insights: 'Perspectivas', settings: 'Configuración', clinicianHome: 'Inicio', clinicianDashboard: 'Panel', patients: 'Pacientes', patientCount: '{count} pacientes', searchPatients: 'Buscar pacientes', timeInRangeShort: 'TIR {percent}%', syncedToday: 'sincronizado hoy', lowRangeReview: 'Revisar rango bajo', highRangeReview: 'Revisar rango alto', noPatientsFound: 'No se encontraron pacientes', openPatientProfile: 'Abrir perfil del paciente', patientProfile: 'Perfil del paciente', yearsOld: '{age} años', bmi: 'IMC', pregnant: 'Embarazada', hba1c: 'HbA1c', sevenDaySummary: 'Informe resumido de 7 días', summaryReportDescription: 'Datos de glucosa registrados durante los últimos siete días.', averageGlucose: 'Glucosa promedio', recordedMeals: 'Comidas registradas', latestDayAverage: 'Promedio del último día', 'Type 1': 'Tipo 1', 'Type 2': 'Tipo 2', 'Asian': 'Asiática', 'Hispanic or Latino': 'Hispana o latina',
    todayDescription: 'Aquí aparecerá el pronóstico de tus comidas y glucosa.', personalizationStatus: 'Estado de personalización', modelReady: 'Tu modelo está listo para predicciones de comidas.', daysCollected: '{days} de {required} días recopilados. Las predicciones se habilitarán cuando tu línea de base esté lista.', checkingStatus: 'Comprobando el estado de personalización…', mealPhoto: 'Foto de comida', addPhotoDescription: 'Agrega una foto para capturar los detalles de tu comida.', addMealPhoto: 'Agregar foto de comida', preparingPhoto: 'Preparando tu foto…',
    healthQuestionnaire: 'Cuestionario de salud', skip: 'Omitir', back: 'Atrás', continue: 'Continuar', finish: 'Finalizar', aboutYou: 'Sobre ti', diabetesHistory: 'Historial de diabetes', tellUs: 'Cuéntanos sobre tu salud', questionnaireDescription: 'Aproximadamente 5 minutos. Tus respuestas personalizan las predicciones de glucosa durante esta sesión.', diabetesDescription: 'Esto ayuda a personalizar tus perspectivas y predicciones de glucosa.', age: 'Edad', height: 'Estatura', weight: 'Peso', gender: 'Género', pregnancy: '¿Estás embarazada actualmente?', pregnancyDescription: 'El embarazo cambia los objetivos de glucosa. Tu equipo de atención ajustará las recomendaciones.', ethnicity: 'Origen étnico', selectAll: 'Selecciona todas las opciones que correspondan.', diabetesType: 'Tipo de diabetes', diabetesManagement: '¿Cómo manejas la diabetes?', notProvided: 'No proporcionado', management: 'Manejo', sessionOnly: 'SOLO SESIÓN', editHealthProfile: 'Editar perfil de salud', saveChanges: 'Guardar cambios', backToSettings: 'Volver a configuración',
    addMeal: 'Agregar comida', mealType: 'Tipo de comida', mealName: 'Nombre de la comida', carbohydrates: 'Carbohidratos', mealTime: 'Hora de la comida', photo: 'Foto', nutritionEstimation: 'Se usa para la estimación nutricional', replacePhoto: 'Reemplazar foto', optionalPrediction: 'Opcional para esta predicción de demostración.', predictResponse: 'Predecir respuesta de glucosa', predictionPersonalization: 'Usa tus últimos 7 días de datos de CGM, frecuencia cardíaca y medicación para personalizar la predicción.', enterMealName: 'Ingresa el nombre de una comida antes de crear una predicción.', predictionUnavailable: 'No se pudo crear una predicción. Inténtalo de nuevo.',
    photoCancelled: 'Se canceló la selección de foto.', photoPermissionDenied: 'Se denegó el acceso a la cámara. Habilítalo en Configuración para tomar una foto de comida.', photoNormalizationError: 'No pudimos preparar esa foto. Prueba con otra imagen.', photoOversize: 'La foto supera los 8 MB después de la preparación. Prueba con otra imagen.', currentTimePreview: 'Usa la hora actual en la vista previa web.', discardMealDraft: 'Descartar borrador de comida y volver al inicio', mealNamePlaceholder: 'p. ej., avena con fruta',
    historyFootnote: 'Cada día usa muestras registradas de CGMacros; los detalles de las comidas están disponibles para el día final.', glucoseHeartRate: 'Tu historial de glucosa y frecuencia cardíaca para {day}', hoursSince: 'Horas desde que comenzó el día de muestra.', glucose: 'Glucosa', heartRate: 'Frecuencia cardíaca', dailyAverage: 'Promedio diario de glucosa (mg/dL)', loggedMeals: 'Comidas registradas · Día {day}', carbs: 'g de carbohidratos', peak: 'pico mg/dL', timeInRange: 'Tiempo en rango · últimos 7 días', low: 'Bajo <70', inRange: 'En rango', high: 'Alto >140', observedMealResponse: 'Respuesta observada a la comida · primeros 7 días', meanGlucoseChange: 'Cambio medio observado de glucosa', meanRise: 'Aumento medio observado a los 60 min: +{value} mg/dL en {count} ventanas completas de respuesta.', recordedCoverage: 'Cobertura de datos registrados', coverageText: '{days} días de CGM y {meals} comidas registradas en la línea de base mostrada.', responseText: '{count} comidas tuvieron ventanas completas de glucosa de 0–120 minutos.', mealPrediction: 'Predicción de comida', predictedObserved: 'Predicho · observado', glucoseAfterMeal: 'Glucosa después de la comida', confidence: 'Confianza', goHome: 'Volver a la página de inicio', peakLabel: 'Pico', timeToPeak: 'Tiempo hasta el pico', predicted: 'Predicho', observed: 'Observado', uncertainty: 'Incertidumbre', modelPeak: 'Pico del modelo', topDrivers: 'Factores principales de esta predicción',
    Breakfast: 'Desayuno', Lunch: 'Almuerzo', Dinner: 'Cena', Snack: 'Refrigerio', Female: 'Mujer', Male: 'Hombre', 'Non-binary': 'No binario', 'Prefer not to say': 'Prefiero no decirlo', Yes: 'Sí', No: 'No', 'Not sure': 'No estoy seguro/a',
    signOut: 'Cerrar sesión',
    signOutBody: 'Necesitarás tu correo electrónico y contraseña para volver a acceder a Nutrix.',
    signOutTitle: '¿Cerrar sesión?',
    seniorMode: 'Modo para personas mayores', seniorModeDescription: 'Usa colores de mayor contraste, texto más grande y controles más grandes.',
  },
  'zh-Hans': {
    accountFallback: '您的账户',
    cancel: '取消',
    healthProfile: '健康档案',
    healthProfileDescription: '查看和编辑您的问卷答案',
    language: '语言',
    languageDescription: '选择 Nutrix 的语言',
    preferences: '您的偏好设置',
    preferencesDescription: '饮食偏好、血糖数据、通知和隐私设置将显示在这里。',
    today: '今日', history: '历史', insights: '洞察', settings: '设置', clinicianHome: '首页', clinicianDashboard: '控制面板', patients: '患者', patientCount: '{count} 位患者', searchPatients: '搜索患者', timeInRangeShort: '目标范围 {percent}%', syncedToday: '今日已同步', lowRangeReview: '请复查低血糖范围', highRangeReview: '请复查高血糖范围', noPatientsFound: '未找到患者', openPatientProfile: '打开患者档案', patientProfile: '患者档案', yearsOld: '{age} 岁', bmi: 'BMI', pregnant: '是否怀孕', hba1c: '糖化血红蛋白', sevenDaySummary: '7 天汇总报告', summaryReportDescription: '最近 7 天的已记录血糖数据。', averageGlucose: '平均血糖', recordedMeals: '已记录餐食', latestDayAverage: '最近一天平均值', 'Type 1': '1 型', 'Type 2': '2 型', 'Asian': '亚洲人', 'Hispanic or Latino': '西班牙裔或拉丁裔',
    todayDescription: '您的餐食和血糖预测将在这里显示。', personalizationStatus: '个性化状态', modelReady: '您的模型已准备好进行餐食预测。', daysCollected: '已收集 {days} / {required} 天。基线数据准备好后将解锁预测。', checkingStatus: '正在检查个性化状态…', mealPhoto: '餐食照片', addPhotoDescription: '添加照片以记录餐食详情。', addMealPhoto: '添加餐食照片', preparingPhoto: '正在准备您的照片…',
    healthQuestionnaire: '健康问卷', skip: '跳过', back: '返回', continue: '继续', finish: '完成', aboutYou: '关于您', diabetesHistory: '糖尿病病史', tellUs: '告诉我们您的健康状况', questionnaireDescription: '约需 5 分钟。您的回答会在本次会话中个性化血糖预测。', diabetesDescription: '这有助于个性化您的血糖洞察和预测。', age: '年龄', height: '身高', weight: '体重', gender: '性别', pregnancy: '您目前是否怀孕？', pregnancyDescription: '怀孕会改变血糖目标。您的护理团队将调整相关建议。', ethnicity: '族裔', selectAll: '选择所有适用项', diabetesType: '糖尿病类型', diabetesManagement: '您如何管理糖尿病？', notProvided: '未提供', management: '管理方式', sessionOnly: '仅限本次会话', editHealthProfile: '编辑健康档案', saveChanges: '保存更改', backToSettings: '返回设置',
    addMeal: '添加餐食', mealType: '餐食类型', mealName: '餐食名称', carbohydrates: '碳水化合物', mealTime: '用餐时间', photo: '照片', nutritionEstimation: '用于营养估算', replacePhoto: '替换照片', optionalPrediction: '此演示预测为可选项。', predictResponse: '预测血糖反应', predictionPersonalization: '使用您过去 7 天的 CGM、心率和用药数据来个性化预测。', enterMealName: '请在创建预测前输入餐食名称。', predictionUnavailable: '无法创建预测，请重试。',
    photoCancelled: '已取消照片选择。', photoPermissionDenied: '相机访问被拒绝。请在“设置”中启用它以拍摄餐食照片。', photoNormalizationError: '无法处理该照片，请尝试另一张图片。', photoOversize: '处理后照片超过 8 MB，请尝试另一张图片。', currentTimePreview: '网页预览使用当前时间。', discardMealDraft: '丢弃餐食草稿并返回首页', mealNamePlaceholder: '例如：水果燕麦粥',
    historyFootnote: '每天使用已记录的 CGMacros 样本；餐食详情可在最后一个样本日查看。', glucoseHeartRate: '{day} 的血糖和心率历史', hoursSince: '自样本日开始以来的小时数。', glucose: '血糖', heartRate: '心率', dailyAverage: '每日平均血糖 (mg/dL)', loggedMeals: '已记录餐食 · 第 {day} 天', carbs: '克碳水化合物', peak: '峰值 mg/dL', timeInRange: '目标范围内时间 · 最近 7 天', low: '低 <70', inRange: '范围内', high: '高 >140', observedMealResponse: '观察到的餐后反应 · 前 7 天', meanGlucoseChange: '平均观察到的血糖变化', meanRise: '60 分钟平均观察到的升高：+{value} mg/dL，来自 {count} 个完整餐后窗口。', recordedCoverage: '已记录数据覆盖范围', coverageText: '显示的基线中有 {days} 天 CGM 和 {meals} 次记录餐食。', responseText: '{count} 餐具有完整的 0–120 分钟血糖窗口用于反应计算。', mealPrediction: '餐食预测', predictedObserved: '预测值 · 观察值', glucoseAfterMeal: '餐后血糖', confidence: '置信度', goHome: '返回首页', peakLabel: '峰值', timeToPeak: '达峰时间', predicted: '预测值', observed: '观察值', uncertainty: '不确定性', modelPeak: '模型峰值', topDrivers: '此预测的主要因素',
    Breakfast: '早餐', Lunch: '午餐', Dinner: '晚餐', Snack: '加餐', Female: '女性', Male: '男性', 'Non-binary': '非二元性别', 'Prefer not to say': '不愿透露', Yes: '是', No: '否', 'Not sure': '不确定',
    signOut: '退出登录',
    signOutBody: '您需要使用电子邮箱和密码再次登录 Nutrix。',
    signOutTitle: '退出登录？',
    seniorMode: '长者模式', seniorModeDescription: '使用高对比度颜色、更大的文字和更大的控件。',
  },
};

type LanguageContextValue = {
  language: LanguageCode;
  setLanguage: (language: LanguageCode) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
};

const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: PropsWithChildren) {
  const [language, setLanguage] = useState<LanguageCode>('en');

  const value = useMemo(
    () => ({
      language,
      setLanguage,
      t: (key: string, values = {}) => Object.entries(values).reduce((text, [name, value]) => text.replaceAll(`{${name}}`, String(value)), translations[language][key] ?? translations.en[key] ?? key),
    }),
    [language],
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const value = useContext(LanguageContext);

  if (!value) {
    throw new Error('useLanguage must be used inside LanguageProvider.');
  }

  return value;
}
