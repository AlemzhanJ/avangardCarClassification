'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Logo from '../components/Logo';
import { inferSeverity } from "../lib/severity";
import { ChartBarHorizontal, Sparkle } from '@phosphor-icons/react';

interface ClassificationResult {
  label: string;
  probability: number;
  severity: 'Low' | 'Medium' | 'High';
  severityPercentage: number;
  details?: {
    predicted_class: string;
    damage_type: string;
    probabilities: Record<string, number>;
  };
}

// Language translations
const translations = {
  En: {
    carClassificationSystem: 'Car Classification System',
    aiCarConditionAnalysis: 'AI Car Condition Analysis',
    footerText: '© 2025 Avangard Car Classification',
    heroJoinA: 'Join a',
    heroSmartJourney: 'smart journey',
    heroWhereAI: 'Where AI analyzes',
    heroCarConditions: 'car conditions',
    uploadDescription: 'Upload a car image to analyze cleanliness and damage conditions using advanced machine learning models',
    clickToUpload: 'Click to upload car image',
    fileTypes: 'PNG, JPG up to 10MB',
    hideLicensePlates: 'Hide license plates',
    analyzeImage: 'Analyze Image',
    processing: 'Processing...',
    imagePreview: 'Image Preview',
    licensePlatesHidden: 'License plates hidden',
    cleanlinessAnalysis: 'Cleanliness Analysis',
    integrityAnalysis: 'Integrity Analysis',
    status: 'Status:',
    confidence: 'Confidence:',
    dirtLevel: 'Dirt Level:',
    damageLevel: 'Damage Level:',
    uploadAnalyzeMessage: 'Upload and analyze an image to see results',
    cleanlinessDetection: 'Cleanliness Detection',
    cleanlinessDescription: 'AI-powered analysis to determine if the car is clean or dirty with severity levels',
    damageAssessment: 'Damage Assessment',
    damageDescription: 'Detect scratches, dents, and other damage with detailed severity analysis',
    privacyProtection: 'Privacy Protection',
    privacyDescription: 'Data is not sent or stored anywhere.',
    footerText: '© 2025 inDrive Car Classification System. Advanced AI for automotive assessment.',
    clean: 'Clean',
    dirty: 'Dirty',
    damaged: 'Damaged',
    undamaged: 'Undamaged',
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    predictedClass: 'Predicted class',
    damageType: 'Damage type',
    classProbabilities: 'Class probabilities',
    damageScratch: 'Scratch',
    damageDent: 'Dent'
  },
  Ru: {
    carClassificationSystem: 'Система классификации автомобилей',
    aiCarConditionAnalysis: 'ИИ Анализ состояния автомобиля',
    heroJoinA: 'Присоединяйтесь к',
    heroSmartJourney: 'умному путешествию',
    heroWhereAI: 'Где ИИ анализирует',
    heroCarConditions: 'состояние автомобилей',
    uploadDescription: 'Загрузите изображение автомобиля для анализа чистоты и повреждений с помощью передовых моделей машинного обучения',
    clickToUpload: 'Нажмите для загрузки изображения авто',
    fileTypes: 'PNG, JPG до 10МБ',
    hideLicensePlates: 'Скрыть номерные знаки',
    analyzeImage: 'Анализировать изображение',
    processing: 'Обработка...',
    imagePreview: 'Предпросмотр изображения',
    licensePlatesHidden: 'Номерные знаки скрыты',
    cleanlinessAnalysis: 'Анализ чистоты',
    integrityAnalysis: 'Анализ целостности',
    status: 'Статус:',
    confidence: 'Уверенность:',
    dirtLevel: 'Уровень загрязнения:',
    damageLevel: 'Уровень повреждений:',
    uploadAnalyzeMessage: 'Загрузите и проанализируйте изображение для просмотра результатов',
    cleanlinessDetection: 'Определение чистоты',
    cleanlinessDescription: 'ИИ-анализ для определения чистоты или загрязненности автомобиля с уровнями серьезности',
    damageAssessment: 'Оценка повреждений',
    damageDescription: 'Обнаружение царапин, вмятин и других повреждений с детальным анализом серьезности',
    privacyProtection: 'Защита конфиденциальности',
    privacyDescription: 'Опциональное размытие номерных знаков для защиты конфиденциальности во время анализа',
    footerText: '© 2025 inDrive Система классификации автомобилей. Передовой ИИ для автомобильной оценки.',
    clean: 'Чистый',
    dirty: 'Грязный',
    damaged: 'Повреждён',
    undamaged: 'Не повреждён',
    low: 'Низкий',
    medium: 'Средний',
    high: 'Высокий',
    predictedClass: 'Предсказанный класс',
    damageType: 'Тип повреждения',
    classProbabilities: 'Вероятности классов',
    damageScratch: 'Царапина',
    damageDent: 'Вмятина'
  },
  Kz: {
    carClassificationSystem: 'Автомобиль жіктеу жүйесі',
    aiCarConditionAnalysis: 'ЖИ Автомобиль жай-күйін талдау',
    heroJoinA: 'Қосылыңыз',
    heroSmartJourney: 'ақылды саяхатқа',
    heroWhereAI: 'Мұнда ЖИ талдайды',
    heroCarConditions: 'автомобиль жай-күйін',
    uploadDescription: 'Озық машиналық оқыту модельдерін пайдалана отырып, тазалық пен зақымдануды талдау үшін автомобиль суретін жүктеңіз',
    clickToUpload: 'Автомобиль суретін жүктеу үшін басыңыз',
    fileTypes: 'PNG, JPG 10МБ дейін',
    hideLicensePlates: 'Нөмірлік белгілерді жасыру',
    analyzeImage: 'Суретті талдау',
    processing: 'Өңдеу...',
    imagePreview: 'Сурет алдын ала қарау',
    licensePlatesHidden: 'Нөмірлік белгілер жасырылған',
    cleanlinessAnalysis: 'Тазалық талдауы',
    integrityAnalysis: 'Тұтастық талдауы',
    status: 'Күйі:',
    confidence: 'Сенімділік:',
    dirtLevel: 'Ластану деңгейі:',
    damageLevel: 'Зақымдану деңгейі:',
    uploadAnalyzeMessage: 'Нәтижелерді көру үшін суретті жүктеп талдаңыз',
    cleanlinessDetection: 'Тазалықты анықтау',
    cleanlinessDescription: 'Автомобильдің таза немесе лас екенін қатаңдық деңгейлерімен анықтайтын ЖИ-талдау',
    damageAssessment: 'Зақымдануды бағалау',
    damageDescription: 'Сызаттар, ойықтар және басқа зақымданулар мен толық қатаңдық талдауын анықтау',
    privacyProtection: 'Құпиялылықты қорғау',
    privacyDescription: 'Талдау кезінде құпиялылықты қорғау үшін нөмірлік белгілерді қосымша бұлдырлау',
    footerText: '© 2025 inDrive Автомобиль жіктеу жүйесі. Автомобиль бағалауы үшін озық ЖИ.',
    clean: 'Таза',
    dirty: 'Лас',
    damaged: 'Зақымданған',
    undamaged: 'Зақымданбаған',
    low: 'Төмен',
    medium: 'Орта',
    high: 'Жоғары',
    predictedClass: 'Болжанған класс',
    damageType: 'Зақым түрі',
    classProbabilities: 'Сынып ықтималдықтары',
    damageScratch: 'Scratch',
    damageDent: 'Dent'
  }
};

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  // Cleanliness is not used currently; simplified UI without this state
  const [integrityResult, setIntegrityResult] = useState<ClassificationResult | null>(null);
  const [currentLanguage, setCurrentLanguage] = useState<'En' | 'Ru' | 'Kz'>('En');
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);

  const languages = [
    { code: 'En' as const, name: 'English' },
    { code: 'Ru' as const, name: 'Русский' },
    { code: 'Kz' as const, name: 'Қазақша' }
  ];

  // Load language from localStorage on mount
  useEffect(() => {
    const savedLanguage = localStorage.getItem('selectedLanguage') as 'En' | 'Ru' | 'Kz';
    if (savedLanguage && ['En', 'Ru', 'Kz'].includes(savedLanguage)) {
      setCurrentLanguage(savedLanguage);
    }
  }, []);

  const handleLanguageChange = (langCode: 'En' | 'Ru' | 'Kz') => {
    setCurrentLanguage(langCode);
    localStorage.setItem('selectedLanguage', langCode);
    setShowLanguageDropdown(false);
  };

  const t = translations[currentLanguage];

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        // reset integrity result on new image
        setIntegrityResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;
    setIsProcessing(true);

    try {
      // Run severity model for damage assessment
      const sev = await inferSeverity(selectedImage);

      // Map severity label
      const severityMap: Record<string, 'Low' | 'Medium' | 'High'> = {
        low: 'Low',
        med: 'Medium',
        high: 'High'
      };

      setIntegrityResult({
        label: t.damaged,
        probability: sev.confidence,
        severity: severityMap[sev.severity] ?? 'Medium',
        severityPercentage: Math.round(sev.confidence * 100),
        details: {
          predicted_class: sev.predicted_class,
          damage_type: sev.damage_type,
          probabilities: sev.probabilities,
        },
      });

      // Cleanliness flow is not implemented yet
    } catch (err) {
      console.error('Severity inference failed', err);
      // fallback UI
      setIntegrityResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Low': return 'text-green-600 dark:text-green-400';
      case 'Medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'High': return 'text-red-600 dark:text-red-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getSeverityText = (severity: string) => {
    switch (severity) {
      case 'Low': return t.low;
      case 'Medium': return t.medium;
      case 'High': return t.high;
      default: return severity;
    }
  };

  const getDamageTypeText = (type: string) => {
    if (type === 'scratch') return t.damageScratch || 'Scratch';
    if (type === 'dent') return t.damageDent || 'Dent';
    return type;
  };

  // Map model class key like "scratch_high" -> localized human text, e.g. "Царапина — Высокий"
  const formatClassLabel = (key: string) => {
    const parts = key.split('_');
    if (parts.length !== 2) return key;
    const [type, sev] = parts as [string, 'low' | 'med' | 'high'];
    const sevMap: Record<string, 'Low' | 'Medium' | 'High'> = { low: 'Low', med: 'Medium', high: 'High' };
    return `${getDamageTypeText(type)} — ${getSeverityText(sevMap[sev] ?? 'Medium')}`;
  };

  // Only keep icon for probabilities as requested
  const IconProbs = () => (<ChartBarHorizontal size={18} weight="bold" />);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-card-border bg-card-background">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Logo />
            <div className="flex items-center gap-4">
              {/* Language Selector */}
              <div className="relative">
                <button
                  onClick={() => setShowLanguageDropdown(!showLanguageDropdown)}
                  className="flex items-center gap-2 p-2 rounded-full border border-gray-300 dark:border-gray-400 hover:border-indrive-green transition-colors bg-white dark:bg-background"
                >
                  <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                  </svg>
                  <span className="text-sm font-medium">{currentLanguage}</span>
                  <svg 
                    className={`w-4 h-4 transition-transform ${showLanguageDropdown ? 'rotate-180' : ''}`} 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                
                {/* Dropdown */}
                {showLanguageDropdown && (
                  <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-background border border-gray-200 dark:border-gray-400 rounded-lg shadow-lg z-50">
                    <div className="p-2">
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 px-3 py-2 border-b border-gray-100 dark:border-gray-400">
                        Country and language
                      </div>
                      {languages.map((lang) => (
                        <button
                          key={lang.code}
                          onClick={() => handleLanguageChange(lang.code)}
                          className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                            currentLanguage === lang.code ? 'bg-indrive-green text-black font-medium' : 'text-gray-700 dark:text-gray-300'
                          }`}
                        >
                          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                          </svg>
                          <span>{lang.name}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {/* Hero Section */}
        <div className="text-center mb-6 py-8">
          <h1 className="text-4xl sm:text-5xl font-black mb-6 tracking-tight text-foreground">
            <span className="inline-block">
              {t.heroJoinA}{' '}
              <span className="relative inline-block font-black">
                <span className="absolute inset-0 bg-indrive-green transform -rotate-1 translate-x-1 translate-y-0.5 px-2 py-1"></span>
                <span className="absolute inset-0 bg-indrive-green transform rotate-0.5 -translate-x-0.5 translate-y-1 px-2 py-1"></span>
                <span className="relative z-10 text-black px-2 py-1">
                  {t.heroSmartJourney}
                </span>
              </span>
            </span>
            <br />
            <span className="text-foreground">
              {t.heroWhereAI}
            </span>
            <br />
            <span className="text-foreground">
              {t.heroCarConditions}
            </span>
          </h1>
          <p className="text-xl font-medium text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            {t.uploadDescription}
          </p>
        </div>

        {/* Upload Section (image replaces drop area) */}
        <div className="max-w-2xl mx-auto mb-4">
          <div className={`${selectedImage ? 'border border-gray-200 dark:border-gray-700' : 'border-2 border-dashed border-gray-300 dark:border-gray-600'} rounded-lg overflow-hidden relative`}>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
              disabled={selectedImage !== null}
            />
            <label htmlFor={selectedImage ? undefined : "image-upload"} className={`block ${selectedImage ? '' : 'cursor-pointer'}`}>
              {selectedImage ? (
                <div className="relative">
                  <Image
                    src={selectedImage}
                    alt="Selected"
                    className="w-full h-auto"
                    width={1200}
                    height={800}
                    unoptimized
                  />
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      setSelectedImage(null);
                      setIntegrityResult(null);
                    }}
                    className="absolute top-0 right-0 w-8 h-8 bg-indrive-green hover:bg-indrive-green-dark rounded-full flex items-center justify-center shadow-lg transition-colors z-10"
                    title="Remove image"
                  >
                    <svg className="w-4 h-4 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ) : (
                <div className="p-8 text-center hover:border-indrive-green transition-colors">
                  <div className="mb-4">
                    <svg className="w-12 h-12 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                    {t.clickToUpload}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {t.fileTypes}
                  </p>
                </div>
              )}
            </label>
          </div>
        </div>

        {/* Analyze button */}
        {selectedImage && (
          <div className="max-w-2xl mx-auto mb-6 flex items-center justify-center">
            <button
              onClick={processImage}
              disabled={isProcessing}
              className="bg-indrive-green hover:bg-indrive-green-dark text-black font-medium px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              <Sparkle size={16} weight="bold" />
              {isProcessing ? t.processing : t.analyzeImage}
            </button>
          </div>
        )}

        {/* Results (compact, no cards) */}
        {selectedImage && integrityResult && (
          <div className="max-w-2xl mx-auto space-y-4 mb-10">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="flex items-center justify-between gap-3 p-3 border border-card-border rounded-md bg-card-background">
                <div className="text-gray-600 dark:text-gray-300 font-medium">{t.status}</div>
                <div className="font-semibold text-red-600 dark:text-red-400">{integrityResult.label}</div>
              </div>
              <div className="flex items-center justify-between gap-3 p-3 border border-card-border rounded-md bg-card-background">
                <div className="text-gray-600 dark:text-gray-300 font-medium">{t.confidence}</div>
                <div className="font-semibold">{(integrityResult.probability * 100).toFixed(1)}%</div>
              </div>
              <div className="flex items-center justify-between gap-3 p-3 border border-card-border rounded-md bg-card-background sm:col-span-2">
                <div className="text-gray-600 dark:text-gray-300 font-medium">{t.damageLevel}</div>
                <div className={`font-semibold ${getSeverityColor(integrityResult.severity)}`}>{getSeverityText(integrityResult.severity)} ({integrityResult.severityPercentage}%)</div>
              </div>
            </div>

            {integrityResult.details && (
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="flex items-center justify-between gap-3 p-3 border border-card-border rounded-md bg-card-background">
                    <div className="text-gray-600 dark:text-gray-300 font-medium">{t.predictedClass}</div>
                    <div className="font-semibold text-right">{formatClassLabel(integrityResult.details.predicted_class)}</div>
                  </div>
                  <div className="flex items-center justify-between gap-3 p-3 border border-card-border rounded-md bg-card-background">
                    <div className="text-gray-600 dark:text-gray-300 font-medium">{t.damageType}</div>
                    <div className="font-semibold">{getDamageTypeText(integrityResult.details.damage_type)}</div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-2 text-gray-700 dark:text-gray-300"><IconProbs /> <span className="font-medium">{t.classProbabilities}</span></div>
                  <div className="space-y-2">
                    {Object.entries(integrityResult.details.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, p]) => (
                        <div key={label} className="flex items-center gap-3">
                          <div className="w-44 text-xs text-gray-600 dark:text-gray-400 truncate">{formatClassLabel(label)}</div>
                          <div className="flex-1 bg-gray-200 dark:bg-gray-700 h-2 rounded-full">
                            <div className="bg-indrive-green h-2 rounded-full" style={{ width: `${(p * 100).toFixed(1)}%` }}></div>
                          </div>
                          <div className="w-12 text-xs text-right">{(p * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Features Section */}
        {!selectedImage && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto mt-16">
            <div className="text-center p-6">
              <div className="w-12 h-12 bg-indrive-green rounded-lg mx-auto mb-4 flex items-center justify-center">
                <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold mb-2">{t.cleanlinessDetection}</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {t.cleanlinessDescription}
              </p>
            </div>

            <div className="text-center p-6">
              <div className="w-12 h-12 bg-indrive-green rounded-lg mx-auto mb-4 flex items-center justify-center">
                <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold mb-2">{t.damageAssessment}</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {t.damageDescription}
              </p>
            </div>

            <div className="text-center p-6">
              <div className="w-12 h-12 bg-indrive-green rounded-lg mx-auto mb-4 flex items-center justify-center">
                <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold mb-2">{t.privacyProtection}</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {t.privacyDescription}
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-card-border bg-card-background mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col items-center justify-center gap-2">
            <Logo width={120} height={40} />
            <p className="text-gray-600 dark:text-gray-400 text-sm">{t.footerText}</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
