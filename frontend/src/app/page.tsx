'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Logo from '../components/Logo';

interface ClassificationResult {
  label: string;
  probability: number;
  severity: 'Low' | 'Medium' | 'High';
  severityPercentage: number;
}

// Language translations
const translations = {
  En: {
    carClassificationSystem: 'Car Classification System',
    aiCarConditionAnalysis: 'AI Car Condition Analysis',
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
    privacyDescription: 'Optional license plate blurring to protect privacy during analysis',
    footerText: '© 2024 inDrive Car Classification System. Advanced AI for automotive assessment.',
    clean: 'Clean',
    dirty: 'Dirty',
    damaged: 'Damaged',
    undamaged: 'Undamaged',
    low: 'Low',
    medium: 'Medium',
    high: 'High'
  },
  Ru: {
    carClassificationSystem: 'Система классификации автомобилей',
    aiCarConditionAnalysis: 'ИИ Анализ состояния автомобиля',
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
    footerText: '© 2024 inDrive Система классификации автомобилей. Передовой ИИ для автомобильной оценки.',
    clean: 'Чистый',
    dirty: 'Грязный',
    damaged: 'Повреждён',
    undamaged: 'Не повреждён',
    low: 'Низкий',
    medium: 'Средний',
    high: 'Высокий'
  },
  Kz: {
    carClassificationSystem: 'Автомобиль жіктеу жүйесі',
    aiCarConditionAnalysis: 'ЖИ Автомобиль жай-күйін талдау',
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
    footerText: '© 2024 inDrive Автомобиль жіктеу жүйесі. Автомобиль бағалауы үшін озық ЖИ.',
    clean: 'Таза',
    dirty: 'Лас',
    damaged: 'Зақымданған',
    undamaged: 'Зақымданбаған',
    low: 'Төмен',
    medium: 'Орта',
    high: 'Жоғары'
  }
};

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hideNumbers, setHideNumbers] = useState(false);
  const [cleanlinessResult, setCleanlinessResult] = useState<ClassificationResult | null>(null);
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
        setCleanlinessResult(null);
        setIntegrityResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;
    
    setIsProcessing(true);
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock results for demonstration
    setCleanlinessResult({
      label: t.clean,
      probability: 0.85,
      severity: 'Low',
      severityPercentage: 15
    });
    
    setIntegrityResult({
      label: t.damaged,
      probability: 0.72,
      severity: 'Medium',
      severityPercentage: 65
    });
    
    setIsProcessing(false);
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
                  className="flex items-center gap-2 p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:border-indrive-green transition-colors bg-white dark:bg-background"
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
                  <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-background border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50">
                    <div className="p-2">
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 px-3 py-2 border-b border-gray-100 dark:border-gray-700">
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
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-black mb-6 tracking-tight">
            {t.aiCarConditionAnalysis}
          </h1>
          <p className="text-xl font-medium text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            {t.uploadDescription}
          </p>
        </div>

        {/* Upload Section */}
        <div className="max-w-2xl mx-auto mb-8">
          <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:border-indrive-green transition-colors">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <label htmlFor="image-upload" className="cursor-pointer">
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
            </label>
          </div>
        </div>

        {/* Controls */}
        {selectedImage && (
          <div className="max-w-2xl mx-auto mb-8 flex flex-col sm:flex-row gap-4 items-center justify-center">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={hideNumbers}
                onChange={(e) => setHideNumbers(e.target.checked)}
                className="rounded border-gray-300 text-indrive-green focus:ring-indrive-green"
              />
              <span className="text-sm font-medium">{t.hideLicensePlates}</span>
            </label>
            <button
              onClick={processImage}
              disabled={isProcessing}
              className="bg-indrive-green hover:bg-indrive-green-dark text-black font-medium px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? t.processing : t.analyzeImage}
            </button>
          </div>
        )}

        {/* Image Preview and Results */}
        {selectedImage && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {/* Image Preview */}
            <div className="bg-card-background border border-card-border rounded-lg p-6">
              <h3 className="text-lg font-bold mb-4">{t.imagePreview}</h3>
              <div className="relative">
                <Image
                  src={selectedImage}
                  alt="Car preview"
                  className="w-full h-auto rounded-lg"
                  style={hideNumbers ? { filter: 'blur(2px)' } : {}}
                  width={500}
                  height={300}
                  unoptimized
                />
                {hideNumbers && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="bg-black bg-opacity-50 text-white px-3 py-1 rounded">
                      {t.licensePlatesHidden}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            <div className="space-y-6">
              {/* Cleanliness Card */}
              <div className="bg-card-background border border-card-border rounded-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-3"></div>
                  {t.cleanlinessAnalysis}
                </h3>
                {cleanlinessResult ? (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{t.status}</span>
                      <span className={`font-semibold ${cleanlinessResult.label === t.clean ? 'text-green-600 dark:text-green-400' : 'text-orange-600 dark:text-orange-400'}`}>
                        {cleanlinessResult.label}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{t.confidence}</span>
                      <span className="font-semibold">{(cleanlinessResult.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{t.dirtLevel}</span>
                      <span className={`font-semibold ${getSeverityColor(cleanlinessResult.severity)}`}>
                        {getSeverityText(cleanlinessResult.severity)} ({cleanlinessResult.severityPercentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-indrive-green h-2 rounded-full transition-all duration-500"
                        style={{ width: `${cleanlinessResult.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400">{t.uploadAnalyzeMessage}</p>
                )}
              </div>

              {/* Integrity Card */}
              <div className="bg-card-background border border-card-border rounded-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center">
                  <div className="w-3 h-3 bg-purple-500 rounded-full mr-3"></div>
                  {t.integrityAnalysis}
                </h3>
                {integrityResult ? (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Status:</span>
                      <span className={`font-semibold ${integrityResult.label === 'Undamaged' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {integrityResult.label}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{t.confidence}</span>
                      <span className="font-semibold">{(integrityResult.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Damage Level:</span>
                      <span className={`font-semibold ${getSeverityColor(integrityResult.severity)}`}>
                        {integrityResult.severity} ({integrityResult.severityPercentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-indrive-green h-2 rounded-full transition-all duration-500"
                        style={{ width: `${integrityResult.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400">{t.uploadAnalyzeMessage}</p>
                )}
              </div>
            </div>
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
          <div className="text-center text-gray-600 dark:text-gray-400">
            <p>{t.footerText}</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
