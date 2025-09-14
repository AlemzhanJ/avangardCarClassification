'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Logo from '../components/Logo';
import { inferSeverity } from "../lib/severity";
import { inferCleanliness } from "../lib/cleanliness";
import { inferYolo, YoloDet } from "../lib/yolo";
import ShimmerSkeletonOverlay from "../components/ShimmerSkeletonOverlay";
import { Sparkle, Spinner, Smiley, SmileyMeh, SmileySad, ArrowCounterClockwise, Upload } from '@phosphor-icons/react';
import GaugeRadial from "../components/GaugeRadial";

interface ClassificationResult {
  label: string;
  probability: number;
  severity: 'Low' | 'Medium' | 'High';
  severityPercentage: number;
  probabilities?: { undamaged: number; damaged: number };
}

interface CleanlinessResult {
  label: 'Clean' | 'Dirty';
  probability: number;
  probabilities?: { clean: number; dirty: number };
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
    cleanlinessTitle: 'Cleanliness',
    dirtTitle: 'Dirt Level',
    damageTitle: 'Damage',
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
    damageScratch: 'Сызат',
    damageDent: 'Ойық',
    tryAgain: 'Try Again'
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
    cleanlinessTitle: 'Чистота',
    dirtTitle: 'Загрязнение',
    damageTitle: 'Повреждения',
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
    privacyDescription: 'Данные никуда не отправляются и не сохраняются.',
    footerText: '© 2025 Avangard Классификация автомобилей',
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
    ,
    tryAgain: 'Попробовать ещё раз'
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
    cleanlinessTitle: 'Тазалық',
    dirtTitle: 'Ластану деңгейі',
    damageTitle: 'Зақымдану',
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
    privacyDescription: 'Деректер ешқайда жіберілмейді және сақталмайды.',
    footerText: '© 2025 Avangard Автомобиль жіктеуі',
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
    ,
    tryAgain: 'Қайтадан талдау'
  }
};

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  // Cleanliness first stage result
  const [cleanlinessResult, setCleanlinessResult] = useState<CleanlinessResult | null>(null);
  const [integrityResult, setIntegrityResult] = useState<ClassificationResult | null>(null);
  const [yoloDetections, setYoloDetections] = useState<YoloDet[] | null>(null);
  const [imgNaturalSize, setImgNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [analysisDone, setAnalysisDone] = useState(false);
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
        // reset close cross state removed (unused)
        // reset results on new image
        setCleanlinessResult(null);
        setIntegrityResult(null);
        setYoloDetections(null);
        // extract natural size for overlay scaling
        const src = e.target?.result as string;
        const i = new window.Image();
        i.onload = () => setImgNaturalSize({ w: i.naturalWidth || i.width, h: i.naturalHeight || i.height });
        i.src = src;
        setAnalysisDone(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;
    setIsProcessing(true);

    const MIN_FLOW_MS = 1400; // show skeleton shimmer at least one sweep
    const delay = new Promise((resolve) => setTimeout(resolve, MIN_FLOW_MS));

    try {
      // First stage: cleanliness
      const cleanPromise = inferCleanliness(selectedImage);
      // Start damage model in parallel, but we will await after cleanliness for UX ordering
      const sevPromise = inferSeverity(selectedImage);
      let clean: Awaited<ReturnType<typeof inferCleanliness>> | null = null;
      let sev: Awaited<ReturnType<typeof inferSeverity>> | null = null;
      try {
        clean = await cleanPromise;
      } catch (e) {
        console.error('Cleanliness inference failed', e);
        clean = null;
      }
      // Show cleanliness immediately when ready
      if (clean) {
        setCleanlinessResult({
          label: clean.label === 'dirty' ? 'Dirty' : 'Clean',
          probability: clean.confidence,
          probabilities: { clean: clean.probabilities.clean, dirty: clean.probabilities.dirty },
        });
      }
      // Continue with severity
      try {
        sev = await sevPromise;
      } catch (e) {
        console.error('Severity inference failed', e);
        sev = null;
      }
      // Ensure the liquid animation reaches the bottom
      await delay;

      if (sev) {
        const damageProb = sev.probabilities?.damaged ?? (sev.label === 'damaged' ? sev.confidence : 1 - sev.confidence);
        setIntegrityResult({
          label: sev.label === 'damaged' ? t.damaged : t.undamaged,
          probability: damageProb,
          severity: 'Medium',
          severityPercentage: Math.round(damageProb * 100),
          probabilities: sev.probabilities,
        });
      } else {
        setIntegrityResult(null);
      }

      // YOLO trigger: if either dirt or damage risk >= meh threshold
      const mehThreshold = 0.33;
      const dirtRisk = (clean?.label === 'dirty' ? (clean?.confidence ?? 0) : 1 - (clean?.confidence ?? 0));
      const damageRisk = sev ? (sev.probabilities?.damaged ?? (sev.label === 'damaged' ? sev.confidence : 1 - sev.confidence)) : 0;
      if ((dirtRisk >= mehThreshold) || (damageRisk >= mehThreshold)) {
        try {
          const dets = await inferYolo(selectedImage);
          setYoloDetections(dets);
        } catch (e) {
          console.error('YOLO inference failed', e);
          setYoloDetections(null);
        }
      } else {
        setYoloDetections(null);
      }

      // Mark flow complete (UI switches to results-only view)
      setAnalysisDone(true);
    } catch (err) {
      console.error('Processing flow failed', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // removed unused helpers getSeverityText and getDamageTypeText

  // helper kept for potential future text mapping

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

        {/* Upload Section (image preview stays visible with boxes and debug) */}
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
                    width={imgNaturalSize?.w ?? 1200}
                    height={imgNaturalSize?.h ?? 800}
                    unoptimized
                  />
                  {/* YOLO detection overlays */}
                  {yoloDetections && imgNaturalSize && yoloDetections.length > 0 && (
                    <div className="absolute inset-0 pointer-events-none">
                      {yoloDetections.map((d, idx) => {
                        let labelTranslated = d.className ?? String(d.classId);
                        if (d.className === 'dirt' || d.className === 'dirty') labelTranslated = t.dirty;
                        else if (d.className === 'scratch') labelTranslated = t.damageScratch;
                        else if (d.className === 'dent') labelTranslated = t.damageDent;
                        else if (d.className === 'damaged') labelTranslated = t.damaged;
                        else if (d.className === 'undamaged') labelTranslated = t.undamaged;
                        
                        const leftPct = (d.x1 / imgNaturalSize.w) * 100;
                        const topPct = (d.y1 / imgNaturalSize.h) * 100;
                        const wPct = ((d.x2 - d.x1) / imgNaturalSize.w) * 100;
                        const hPct = ((d.y2 - d.y1) / imgNaturalSize.h) * 100;
                        const label = d.className ?? String(d.classId);
                        const conf = Math.round(d.confidence * 100);
                        return (
                          <div key={idx} style={{ left: `${leftPct}%`, top: `${topPct}%`, width: `${wPct}%`, height: `${hPct}%` }} className="absolute border-2 border-indrive-green box-border">
                            <div className="absolute -top-6 left-0 bg-indrive-green text-black text-xs font-semibold px-2 py-0.5 rounded">
                              {labelTranslated} {conf}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {/* White skeleton shimmer overlay while processing */}
                  <ShimmerSkeletonOverlay active={isProcessing} durationMs={1400} />
                  {!analysisDone && (
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        setSelectedImage(null);
                        setIntegrityResult(null);
                        setYoloDetections(null);
                      }}
                      className="absolute top-0 right-0 w-8 h-8 bg-indrive-green hover:bg-indrive-green-dark rounded-full flex items-center justify-center shadow-lg transition-colors z-10"
                      title="Remove image"
                    >
                      <svg className="w-4 h-4 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              ) : (
                <div className="p-8 text-center hover:border-indrive-green transition-colors">
                  <div className="mb-4">
                    <Upload size={48} weight="bold" className="mx-auto text-gray-400" />
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
        {!analysisDone && selectedImage && (
          <div className="max-w-2xl mx-auto mb-6 flex items-center justify-center">
            <button
              onClick={processImage}
              disabled={isProcessing}
              className="bg-indrive-green hover:bg-indrive-green-dark text-black font-medium px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {isProcessing ? (
                <Spinner size={16} weight="bold" className="animate-spin" />
              ) : (
                <Sparkle size={16} weight="bold" />
              )}
              {isProcessing ? t.processing.replace('...', '') : t.analyzeImage}
            </button>
          </div>
        )}

        {/* Results (compact, no cards) */}
        {analysisDone && (
          <div className="max-w-3xl mx-auto space-y-6 mb-10">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Damage gauge */}
              {integrityResult && (
                <GaugeRadial
                  label={t.damageTitle}
                  value={(integrityResult.severityPercentage || 0) / 100}
                  riskScale
                  Icon={
                    (integrityResult.severityPercentage || 0) / 100 > 0.66 ? SmileySad :
                    (integrityResult.severityPercentage || 0) / 100 > 0.33 ? SmileyMeh :
                    Smiley
                  }
                />
              )}
              {/* Dirt level gauge */}
              {cleanlinessResult && (
                <GaugeRadial
                  label={t.dirtTitle}
                  value={cleanlinessResult.label === 'Dirty' ? cleanlinessResult.probability : 1 - cleanlinessResult.probability}
                  riskScale={true}
                  Icon={
                    (cleanlinessResult.label === 'Dirty' ? cleanlinessResult.probability : 1 - cleanlinessResult.probability) > 0.66 ? SmileySad :
                    (cleanlinessResult.label === 'Dirty' ? cleanlinessResult.probability : 1 - cleanlinessResult.probability) > 0.33 ? SmileyMeh :
                    Smiley
                  }
                />
              )}
            </div>
            <div className="flex items-center justify-center mt-20">
              <button
                onClick={() => {
                  setSelectedImage(null);
                  setCleanlinessResult(null);
                  setIntegrityResult(null);
                  setAnalysisDone(false);
                  setIsProcessing(false);
                }}
                className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-indrive-green text-black font-semibold hover:brightness-110 transition cursor-pointer z-10 relative"
                style={{ pointerEvents: 'auto' }}
              >
                <ArrowCounterClockwise size={18} weight="bold" /> {t.tryAgain}
              </button>
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
          <div className="flex flex-col items-center justify-center gap-2">
            <Logo width={120} height={40} />
            <p className="text-gray-600 dark:text-gray-400 text-sm">{t.footerText}</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
