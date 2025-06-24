"""
AI Toxicity Detector folosind xlm-roberta-base-toxic-x
Detectare multilingvă cu suport pentru română și bypass detection
"""

import re
import asyncio
import logging
import time
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import hashlib

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import numpy as np
from langdetect import detect, LangDetectError
import emoji

from config import get_config

logger = logging.getLogger(__name__)

class ToxicityDetector:
    """Detector de toxicitate folosind xlm-roberta-base-toxic-x"""
    
    def __init__(self):
        self.config = get_config()
        
        # Componente AI
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.device = None
        
        # Cache pentru predicții
        self.prediction_cache: Dict[str, Dict] = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Pattern-uri pentru detectarea bypass-urilor
        self.bypass_patterns = self._load_bypass_patterns()
        
        # Pattern-uri toxice pentru română
        self.romanian_toxic_patterns = self._load_romanian_patterns()
        
        # Statistici
        self.stats = {
            'total_predictions': 0,
            'cached_predictions': 0,
            'model_predictions': 0,
            'pattern_detections': 0,
            'processing_times': [],
            'languages_detected': {}
        }
        
        # Flag pentru inițializare
        self.is_initialized = False
    
    async def initialize(self):
        """Inițializează modelul AI și componentele"""
        if self.is_initialized:
            return
        
        logger.info("🤖 Inițializare detector AI toxicitate...")
        start_time = time.time()
        
        try:
            # Detectează device-ul optim
            self.device = self._detect_optimal_device()
            logger.info(f"📱 Device detectat: {self.device}")
            
            # Încarcă modelul și tokenizer-ul
            await self._load_model_async()
            
            # Warm-up model dacă este configurat
            if self.config.ai.warm_up_model:
                await self._warm_up_model()
            
            # Testează funcționalitatea
            await self._test_functionality()
            
            self.is_initialized = True
            
            load_time = time.time() - start_time
            logger.info(f"✅ Detector AI inițializat în {load_time:.2f} secunde")
            
        except Exception as e:
            logger.error(f"❌ Eroare la inițializarea detector-ului AI: {e}")
            raise
    
    def _detect_optimal_device(self) -> str:
        """Detectează device-ul optim pentru inferență"""
        if self.config.ai.device != 'auto':
            return self.config.ai.device
        
        # Verifică CUDA
        if torch.cuda.is_available() and self.config.ai.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU găsit: {gpu_name} ({gpu_memory:.1f}GB)")
            return "cuda"
        
        # Verifică Apple Metal (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple Metal Performance Shaders disponibil")
            return "mps"
        
        # Fallback la CPU
        logger.info("💻 Folosind CPU pentru inferență")
        return "cpu"
    
    async def _load_model_async(self):
        """Încarcă modelul și tokenizer-ul async"""
        model_name = self.config.ai.model_name
        
        logger.info(f"📥 Încărcare model: {model_name}")
        
        # Rulează încărcarea în thread pool pentru a nu bloca event loop-ul
        loop = asyncio.get_event_loop()
        
        # Încarcă tokenizer
        self.tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                do_lower_case=False
            )
        )
        logger.info("✅ Tokenizer încărcat")
        
        # Încarcă model
        self.model = await loop.run_in_executor(
            None,
            lambda: AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        )
        
        # Mută modelul pe device-ul dorit
        if self.device != "cuda" or not hasattr(self.model, 'device'):
            self.model = self.model.to(self.device)
        
        # Setează modul eval pentru inferență
        self.model.eval()
        
        # Creează pipeline-ul pentru inferență rapidă
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True,
            truncation=True,
            max_length=self.config.ai.max_length
        )
        
        logger.info("✅ Model încărcat și configurat")
    
    async def _warm_up_model(self):
        """Warm-up model cu texte de test"""
        logger.info("🔥 Warm-up model...")
        
        test_texts = [
            "Hello, how are you today?",
            "Salut, ce mai faci?",
            "This is a test message.",
            "Acesta este un mesaj de test."
        ]
        
        for text in test_texts:
            try:
                await self._predict_with_model(text)
            except Exception as e:
                logger.warning(f"⚠️ Eroare la warm-up cu '{text}': {e}")
        
        logger.info("✅ Warm-up completat")
    
    async def _test_functionality(self):
        """Testează funcționalitatea detector-ului"""
        test_cases = [
            ("Hello world", False),
            ("Ești un idiot", True),
            ("Du-te dracului", True),
            ("Mulțumesc pentru ajutor", False)
        ]
        
        logger.info("🧪 Testare funcționalitate...")
        
        for text, expected_toxic in test_cases:
            try:
                result = await self.analyze_text(text, threshold=50.0)
                logger.info(f"Test '{text}': {result['is_toxic']} (scor: {result['toxicity_score']:.1f})")
            except Exception as e:
                logger.error(f"❌ Test eșuat pentru '{text}': {e}")
                raise
        
        logger.info("✅ Toate testele au trecut")
    
    async def analyze_text(self, text: str, threshold: Optional[float] = None, 
                          language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analizează un text pentru toxicitate
        
        Args:
            text: Textul de analizat
            threshold: Pragul de toxicitate (0-100)
            language: Limba textului (opțional, se detectează automat)
        
        Returns:
            Dict cu rezultatele analizei
        """
        if threshold is None:
            threshold = self.config.ai.default_threshold
        
        start_time = time.time()
        self.stats['total_predictions'] += 1
        
        try:
            # Verifică cache-ul
            cache_result = self._check_cache(text, threshold)
            if cache_result:
                self.stats['cached_predictions'] += 1
                self.cache_stats['hits'] += 1
                return cache_result
            
            self.cache_stats['misses'] += 1
            
            # Preprocessing text
            processed_text = self._preprocess_text(text)
            
            # Detectează limba dacă nu este specificată
            if not language:
                language = self._detect_language(processed_text)
            
            # Actualizează statisticile limbilor
            self.stats['languages_detected'][language] = \
                self.stats['languages_detected'].get(language, 0) + 1
            
            # Verifică pattern-urile de bypass și toxicitate
            pattern_result = self._check_patterns(processed_text, language)
            
            # Analiză cu modelul AI
            ai_result = await self._predict_with_model(processed_text)
            
            # Combină rezultatele
            final_result = self._combine_results(
                text, processed_text, pattern_result, ai_result, 
                threshold, language
            )
            
            # Cache-uiește rezultatul
            self._cache_result(text, threshold, final_result)
            
            # Actualizează statisticile
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            if len(self.stats['processing_times']) > 1000:
                self.stats['processing_times'] = self.stats['processing_times'][-1000:]
            
            self.stats['model_predictions'] += 1
            if pattern_result['is_toxic']:
                self.stats['pattern_detections'] += 1
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Eroare la analizarea textului '{text[:50]}...': {e}")
            # Returnează rezultat default în caz de eroare
            return {
                'is_toxic': False,
                'toxicity_score': 0.0,
                'confidence': 0.0,
                'category': 'error',
                'language': language or 'unknown',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessing pentru text înainte de analiză"""
        # Elimină emoji-urile și le convertește la descriere
        text = emoji.demojize(text, language='ro')
        
        # Normalizează unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Elimină spațiile multiple
        text = re.sub(r'\s+', ' ', text)
        
        # Normalizează leetspeak și bypass-uri comune
        text = self._normalize_leetspeak(text)
        
        # Elimină spațierile artificiale între litere
        text = self._remove_spacing_bypass(text)
        
        return text.strip()
    
    def _normalize_leetspeak(self, text: str) -> str:
        """Normalizează leetspeak și substituții comune"""
        substitutions = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', 
            '7': 't', '8': 'b', '@': 'a', '$': 's', '+': 't',
            'ph': 'f', 'gh': 'g', 'ck': 'k', 'qu': 'kw'
        }
        
        result = text.lower()
        for old, new in substitutions.items():
            result = result.replace(old, new)
        
        return result
    
    def _remove_spacing_bypass(self, text: str) -> str:
        """Elimină spațierile artificiale între litere"""
        # Pattern pentru litere separate de spații/punctuație
        pattern = r'\b([a-zA-ZăâîșțĂÂÎȘȚ])\s*[.\-_*]+\s*([a-zA-ZăâîșțĂÂÎȘȚ])'
        
        def replace_spaced(match):
            return match.group(1) + match.group(2)
        
        # Aplică de mai multe ori pentru cuvinte lungi
        for _ in range(3):
            old_text = text
            text = re.sub(pattern, replace_spaced, text)
            if text == old_text:
                break
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """Detectează limba textului"""
        try:
            # Încearcă detectarea automată
            detected = detect(text)
            
            # Verifică dacă e română pe baza cuvintelor cheie
            romanian_indicators = ['să', 'că', 'cu', 'de', 'la', 'în', 'pe', 'și', 'este', 'sunt']
            words = text.lower().split()
            romanian_count = sum(1 for word in words if word in romanian_indicators)
            
            if romanian_count >= 2 or detected == 'ro':
                return 'ro'
            
            return detected if detected in ['en', 'es', 'fr', 'de', 'it'] else 'en'
            
        except (LangDetectError, Exception):
            # Fallback la engleza
            return 'en'
    
    def _check_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Verifică pattern-urile de toxicitate specifice limbii"""
        text_lower = text.lower()
        
        # Pattern-uri pentru română
        if language == 'ro':
            for category, patterns in self.romanian_toxic_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return {
                            'is_toxic': True,
                            'category': category,
                            'confidence': 0.9,
                            'matched_pattern': pattern,
                            'method': 'pattern_matching'
                        }
        
        # Pattern-uri generale de bypass
        for pattern_info in self.bypass_patterns:
            if re.search(pattern_info['pattern'], text_lower):
                return {
                    'is_toxic': True,
                    'category': pattern_info['category'],
                    'confidence': pattern_info['confidence'],
                    'matched_pattern': pattern_info['pattern'],
                    'method': 'bypass_detection'
                }
        
        return {
            'is_toxic': False,
            'category': None,
            'confidence': 0.0,
            'method': 'pattern_matching'
        }
    
    async def _predict_with_model(self, text: str) -> Dict[str, Any]:
        """Predicție cu modelul AI"""
        try:
            # Limitează lungimea textului
            if len(text) > self.config.ai.max_length:
                text = text[:self.config.ai.max_length]
            
            # Rulează predicția în thread pool
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                lambda: self.classifier(text)
            )
            
            # Procesează rezultatul
            if isinstance(result, list) and len(result) > 0:
                scores = result[0]
                
                # Găsește scorul pentru "TOXIC" sau "toxic"
                toxic_score = 0.0
                for score_dict in scores:
                    label = score_dict['label'].upper()
                    if 'TOXIC' in label or label == '1':
                        toxic_score = max(toxic_score, score_dict['score'])
                
                # Convertește la procentaj
                toxicity_score = toxic_score * 100
                
                # Determină categoria pe baza scorurilor
                category = self._determine_category(scores)
                
                return {
                    'toxicity_score': toxicity_score,
                    'confidence': toxic_score,
                    'category': category,
                    'all_scores': scores,
                    'method': 'ai_model'
                }
            
            # Fallback dacă nu găsim rezultat valid
            return {
                'toxicity_score': 0.0,
                'confidence': 0.0,
                'category': 'unknown',
                'method': 'ai_model_fallback'
            }
            
        except Exception as e:
            logger.error(f"❌ Eroare la predicția AI: {e}")
            return {
                'toxicity_score': 0.0,
                'confidence': 0.0,
                'category': 'error',
                'method': 'ai_model_error',
                'error': str(e)
            }
    
    def _determine_category(self, scores: List[Dict]) -> str:
        """Determină categoria de toxicitate pe baza scorurilor"""
        # Mapează label-urile la categorii
        category_mapping = {
            'toxic': 'general',
            'severe_toxic': 'hate_speech',
            'obscene': 'profanity',
            'threat': 'threat',
            'insult': 'harassment',
            'identity_hate': 'discrimination'
        }
        
        best_score = 0.0
        best_category = 'general'
        
        for score_dict in scores:
            label = score_dict['label'].lower()
            score = score_dict['score']
            
            if label in category_mapping and score > best_score:
                best_score = score
                best_category = category_mapping[label]
        
        return best_category
    
    def _combine_results(self, original_text: str, processed_text: str,
                        pattern_result: Dict, ai_result: Dict, 
                        threshold: float, language: str) -> Dict[str, Any]:
        """Combină rezultatele din pattern matching și AI"""
        
        # Dacă pattern-ul a detectat toxicitate cu încredere mare, folosește-l
        if pattern_result['is_toxic'] and pattern_result['confidence'] >= 0.8:
            toxicity_score = 85.0  # Scor mare pentru pattern match
            is_toxic = toxicity_score >= threshold
            confidence = pattern_result['confidence']
            category = pattern_result['category']
            method = pattern_result['method']
        else:
            # Folosește rezultatul AI
            toxicity_score = ai_result['toxicity_score']
            confidence = ai_result['confidence']
            category = ai_result.get('category', 'general')
            method = ai_result['method']
            
            # Combină cu pattern result dacă ambele detectează probleme
            if pattern_result['is_toxic'] and ai_result['toxicity_score'] > 30:
                toxicity_score = min(95.0, toxicity_score * 1.2)  # Boost score
                confidence = max(confidence, pattern_result['confidence'])
        
        is_toxic = toxicity_score >= threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': round(toxicity_score, 2),
            'confidence': round(confidence, 3),
            'category': category,
            'language': language,
            'threshold_used': threshold,
            'method': method,
            'original_text': original_text,
            'processed_text': processed_text,
            'pattern_detected': pattern_result['is_toxic'],
            'ai_score': ai_result['toxicity_score'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _check_cache(self, text: str, threshold: float) -> Optional[Dict]:
        """Verifică cache-ul pentru rezultate existente"""
        if not self.config.ai.cache_predictions:
            return None
        
        cache_key = self._generate_cache_key(text, threshold)
        
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            
            # Verifică dacă cache-ul nu a expirat
            cache_time = datetime.fromisoformat(cached_result['cached_at'])
            if (datetime.utcnow() - cache_time).total_seconds() < self.config.ai.cache_ttl:
                return cached_result['result']
            else:
                # Elimină cache-ul expirat
                del self.prediction_cache[cache_key]
        
        return None
    
    def _cache_result(self, text: str, threshold: float, result: Dict):
        """Cache-uiește rezultatul unei predicții"""
        if not self.config.ai.cache_predictions:
            return
        
        cache_key = self._generate_cache_key(text, threshold)
        
        self.prediction_cache[cache_key] = {
            'result': result,
            'cached_at': datetime.utcnow().isoformat()
        }
        
        # Limitează dimensiunea cache-ului
        if len(self.prediction_cache) > 1000:
            # Elimină cele mai vechi 200 de intrări
            oldest_keys = sorted(
                self.prediction_cache.keys(),
                key=lambda k: self.prediction_cache[k]['cached_at']
            )[:200]
            
            for key in oldest_keys:
                del self.prediction_cache[key]
    
    def _generate_cache_key(self, text: str, threshold: float) -> str:
        """Generează o cheie de cache pentru text și threshold"""
        content = f"{text}|{threshold}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_bypass_patterns(self) -> List[Dict[str, Any]]:
        """Încarcă pattern-urile pentru detectarea bypass-urilor"""
        return [
            {
                'pattern': r'f\s*u\s*c\s*k|s\s*h\s*i\s*t',
                'category': 'profanity',
                'confidence': 0.85
            },
            {
                'pattern': r'b\s*i\s*t\s*c\s*h|w\s*h\s*o\s*r\s*e',
                'category': 'harassment',
                'confidence': 0.80
            },
            {
                'pattern': r'k\s*y\s*s|k\s*i\s*l\s*l.*y\s*o\s*u\s*r\s*s\s*e\s*l\s*f',
                'category': 'threat',
                'confidence': 0.95
            },
            {
                'pattern': r'n\s*i\s*g\s*g\s*e\s*r|n\s*i\s*g\s*g\s*a',
                'category': 'hate_speech',
                'confidence': 0.98
            }
        ]
    
    def _load_romanian_patterns(self) -> Dict[str, List[str]]:
        """Încarcă pattern-urile toxice pentru limba română"""
        return {
            'profanity': [
                r'\bpul[aă]\b', r'\bpizd[aă]\b', r'\bcur\b', r'\bprost[aăiou]*\b',
                r'\bidiot[aăiou]*\b', r'\bcret[iae]n[aăiou]*\b', r'\bproast[aăe]\b'
            ],
            'harassment': [
                r'du-?te\s+(dracului|naibii|la\s+dracu)', r'f[uă]te-?te',
                r'm[âă]-?ta', r't[aă]-?tu', r'ești\s+[aă]?n?\s*prost',
                r'nu\s+vali\s+nimic', r'ești\s+o?\s*nimeni'
            ],
            'threat': [
                r'o\s+s[aă]\s+te\s+(omor|bat|ucid)', r'te\s+(omor|ucid|bat)',
                r'am\s+s[aă]\s+te\s+(omor|bat)', r'm[aă]\s+duc\s+s[aă]\s+te'
            ],
            'hate_speech': [
                r'\btigan[ciou]*\b', r'\bcioara\b', r'\bjidan[ciou]*\b',
                r'\bgay\s+de\s+tot', r'\blezbiana\b', r'\bfag[eiou]*t\b'
            ],
            'discrimination': [
                r'din\s+cauza\s+(culorii|rasei|religiei)',
                r'nu\s+(vă\s+)?plac\s+([a-zA-Z]+ii|[a-zA-Z]+enii)',
                r'toți\s+[a-zA-Z]+ii\s+sunt'
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Returnează statisticile detector-ului"""
        avg_processing_time = 0.0
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        
        cache_hit_rate = 0.0
        total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_cache_requests > 0:
            cache_hit_rate = (self.cache_stats['hits'] / total_cache_requests) * 100
        
        return {
            'model_info': {
                'name': self.config.ai.model_name,
                'device': self.device,
                'initialized': self.is_initialized
            },
            'performance': {
                'total_predictions': self.stats['total_predictions'],
                'cached_predictions': self.stats['cached_predictions'],
                'model_predictions': self.stats['model_predictions'],
                'pattern_detections': self.stats['pattern_detections'],
                'avg_processing_time': round(avg_processing_time, 4),
                'cache_hit_rate': round(cache_hit_rate, 2)
            },
            'cache_stats': self.cache_stats,
            'languages_detected': self.stats['languages_detected'],
            'cache_size': len(self.prediction_cache)
        }
    
    async def cleanup(self):
        """Curăță resursele detector-ului"""
        logger.info("🧹 Curățare detector AI...")
        
        # Curăță cache-ul
        self.prediction_cache.clear()
        
        # Eliberează memoria GPU dacă este folosită
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Setează modelul în modul eval pentru a fi sigur
        if self.model:
            self.model.eval()
        
        logger.info("✅ Detector AI curățat")

# === HELPER FUNCTIONS ===

async def test_detector():
    """Funcție de test pentru detector"""
    detector = ToxicityDetector()
    await detector.initialize()
    
    test_cases = [
        "Hello, how are you?",
        "Ești un prost",
        "Du-te dracului",
        "F u c k you",
        "Mulțumesc pentru ajutor",
        "You're an idiot",
        "I will kill you",
        "Have a nice day!"
    ]
    
    print("🧪 === TESTARE DETECTOR TOXICITATE ===\n")
    
    for text in test_cases:
        result = await detector.analyze_text(text, threshold=50.0)
        
        status = "🔴 TOXIC" if result['is_toxic'] else "🟢 SAFE"
        print(f"{status} | {result['toxicity_score']:5.1f}% | {result['confidence']:.3f} | {text}")
    
    print(f"\n📊 Statistici:")
    stats = detector.get_stats()
    print(f"Predicții totale: {stats['performance']['total_predictions']}")
    print(f"Timp mediu: {stats['performance']['avg_processing_time']:.4f}s")
    print(f"Cache hit rate: {stats['performance']['cache_hit_rate']:.1f}%")
    
    await detector.cleanup()

if __name__ == "__main__":
    asyncio.run(test_detector())