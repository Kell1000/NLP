"""
Advanced NLP Text Analysis Tool
Comprehensive text analysis using NLTK, spaCy, and custom algorithms
"""

import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Note: Install required packages:
# pip install nltk spacy textblob wordcloud matplotlib seaborn
# python -m spacy download en_core_web_sm

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.probability import FreqDist
    from nltk import pos_tag, ne_chunk
    
    # Download required NLTK data
    for package in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with: pip install textblob")


class AdvancedNLPAnalyzer:
    """Comprehensive NLP text analyzer"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive NLP analysis on text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'basic_stats': self._basic_statistics(text),
            'lexical_analysis': self._lexical_analysis(text),
            'readability': self._readability_metrics(text),
            'sentiment': self._sentiment_analysis(text),
            'entities': self._named_entity_recognition(text),
            'keywords': self._keyword_extraction(text),
            'pos_distribution': self._pos_tagging(text),
            'ngrams': self._ngram_analysis(text),
            'technical_terms': self._detect_technical_terms(text),
            'document_structure': self._document_structure(text)
        }
        
        return results
    
    def _basic_statistics(self, text: str) -> Dict:
        """Calculate basic text statistics"""
        # Character counts
        char_count = len(text)
        char_no_spaces = len(text.replace(' ', ''))
        
        # Word tokenization
        if NLTK_AVAILABLE:
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
        else:
            words = text.lower().split()
            sentences = re.split(r'[.!?]+', text)
        
        # Filter out punctuation
        words_clean = [w for w in words if w.isalnum()]
        
        # Unique words
        unique_words = set(words_clean)
        
        # Average lengths
        avg_word_length = np.mean([len(w) for w in words_clean]) if words_clean else 0
        avg_sentence_length = len(words_clean) / len(sentences) if sentences else 0
        
        return {
            'character_count': char_count,
            'character_count_no_spaces': char_no_spaces,
            'word_count': len(words_clean),
            'sentence_count': len(sentences),
            'paragraph_count': len(text.split('\n\n')),
            'unique_word_count': len(unique_words),
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'lexical_diversity': round(len(unique_words) / len(words_clean), 3) if words_clean else 0
        }
    
    def _lexical_analysis(self, text: str) -> Dict:
        """Analyze vocabulary and word usage"""
        if not NLTK_AVAILABLE:
            return {'error': 'NLTK not available'}
        
        words = word_tokenize(text.lower())
        words_clean = [w for w in words if w.isalnum() and w not in self.stop_words]
        
        # Frequency distribution
        freq_dist = FreqDist(words_clean)
        
        # Most common words
        most_common = freq_dist.most_common(20)
        
        # Lemmatization
        lemmatized = [self.lemmatizer.lemmatize(w) for w in words_clean]
        lemma_freq = FreqDist(lemmatized)
        
        # Stemming
        stemmed = [self.stemmer.stem(w) for w in words_clean]
        stem_freq = FreqDist(stemmed)
        
        return {
            'most_common_words': most_common[:10],
            'most_common_lemmas': lemma_freq.most_common(10),
            'vocabulary_size': len(set(words_clean)),
            'hapax_legomena': len([w for w in freq_dist if freq_dist[w] == 1]),  # Words appearing once
            'dis_legomena': len([w for w in freq_dist if freq_dist[w] == 2])  # Words appearing twice
        }
    
    def _readability_metrics(self, text: str) -> Dict:
        """Calculate various readability scores"""
        if not NLTK_AVAILABLE:
            return {'error': 'NLTK not available'}
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_clean = [w for w in words if w.isalnum()]
        
        # Syllable count estimation
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel
            
            if word.endswith('e'):
                syllable_count -= 1
            if syllable_count == 0:
                syllable_count = 1
                
            return syllable_count
        
        total_syllables = sum(count_syllables(w) for w in words_clean)
        
        # Flesch Reading Ease
        if len(sentences) > 0 and len(words_clean) > 0:
            flesch_score = 206.835 - 1.015 * (len(words_clean) / len(sentences)) - 84.6 * (total_syllables / len(words_clean))
        else:
            flesch_score = 0
        
        # Flesch-Kincaid Grade Level
        if len(sentences) > 0 and len(words_clean) > 0:
            fk_grade = 0.39 * (len(words_clean) / len(sentences)) + 11.8 * (total_syllables / len(words_clean)) - 15.59
        else:
            fk_grade = 0
        
        # SMOG Index (requires at least 30 sentences)
        polysyllables = sum(1 for w in words_clean if count_syllables(w) >= 3)
        if len(sentences) >= 30:
            smog_index = 1.0430 * np.sqrt(polysyllables * (30 / len(sentences))) + 3.1291
        else:
            smog_index = None
        
        # Automated Readability Index
        if len(sentences) > 0 and len(words_clean) > 0:
            ari = 4.71 * (len(text.replace(' ', '')) / len(words_clean)) + 0.5 * (len(words_clean) / len(sentences)) - 21.43
        else:
            ari = 0
        
        # Readability interpretation
        def interpret_flesch(score):
            if score >= 90: return "Very Easy (5th grade)"
            elif score >= 80: return "Easy (6th grade)"
            elif score >= 70: return "Fairly Easy (7th grade)"
            elif score >= 60: return "Standard (8th-9th grade)"
            elif score >= 50: return "Fairly Difficult (10th-12th grade)"
            elif score >= 30: return "Difficult (College)"
            else: return "Very Difficult (College graduate)"
        
        return {
            'flesch_reading_ease': round(flesch_score, 2),
            'flesch_interpretation': interpret_flesch(flesch_score),
            'flesch_kincaid_grade': round(fk_grade, 2),
            'smog_index': round(smog_index, 2) if smog_index else 'N/A (need 30+ sentences)',
            'automated_readability_index': round(ari, 2),
            'total_syllables': total_syllables,
            'complex_words': polysyllables
        }
    
    def _sentiment_analysis(self, text: str) -> Dict:
        """Analyze sentiment using multiple approaches"""
        result = {}
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            result['textblob'] = {
                'polarity': round(blob.sentiment.polarity, 3),  # -1 to 1
                'subjectivity': round(blob.sentiment.subjectivity, 3),  # 0 to 1
                'assessment': self._interpret_sentiment(blob.sentiment.polarity)
            }
        
        # Simple lexicon-based sentiment
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'improve', 'success', 'efficient', 'optimize', 'better', 'best',
                         'increase', 'enhance', 'benefit', 'advantage', 'positive', 'effective'}
        
        negative_words = {'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst',
                         'fail', 'failure', 'error', 'problem', 'issue', 'bug',
                         'decrease', 'reduce', 'difficult', 'complex', 'negative', 'risk'}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        result['lexicon_based'] = {
            'positive_words': pos_count,
            'negative_words': neg_count,
            'sentiment_ratio': round((pos_count - neg_count) / len(words), 3) if words else 0
        }
        
        return result
    
    def _interpret_sentiment(self, polarity: float) -> str:
        """Interpret sentiment polarity score"""
        if polarity > 0.3: return "Positive"
        elif polarity < -0.3: return "Negative"
        else: return "Neutral"
    
    def _named_entity_recognition(self, text: str) -> Dict:
        """Extract named entities"""
        entities = {'persons': [], 'organizations': [], 'locations': [], 'dates': [], 'other': []}
        
        if SPACY_AVAILABLE:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ in ['ORG', 'PRODUCT']:
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                else:
                    entities['other'].append(f"{ent.text} ({ent.label_})")
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _pos_tagging(self, text: str) -> Dict:
        """Part-of-speech tagging and distribution"""
        if not NLTK_AVAILABLE:
            return {'error': 'NLTK not available'}
        
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # Count POS distribution
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Simplify POS tags
        pos_categories = defaultdict(int)
        for tag, count in pos_counts.items():
            if tag.startswith('NN'):
                pos_categories['Nouns'] += count
            elif tag.startswith('VB'):
                pos_categories['Verbs'] += count
            elif tag.startswith('JJ'):
                pos_categories['Adjectives'] += count
            elif tag.startswith('RB'):
                pos_categories['Adverbs'] += count
            elif tag in ['DT', 'PDT', 'WDT']:
                pos_categories['Determiners'] += count
            else:
                pos_categories['Other'] += count
        
        return {
            'detailed': dict(pos_counts.most_common(10)),
            'categories': dict(pos_categories)
        }
    
    def _ngram_analysis(self, text: str, n_values: List[int] = [2, 3]) -> Dict:
        """Extract n-grams (bigrams, trigrams, etc.)"""
        if not NLTK_AVAILABLE:
            return {'error': 'NLTK not available'}
        
        words = word_tokenize(text.lower())
        words_clean = [w for w in words if w.isalnum() and w not in self.stop_words]
        
        ngrams_dict = {}
        
        for n in n_values:
            ngrams = list(zip(*[words_clean[i:] for i in range(n)]))
            ngram_freq = Counter(ngrams)
            ngrams_dict[f'{n}-grams'] = [
                (' '.join(gram), count) 
                for gram, count in ngram_freq.most_common(10)
            ]
        
        return ngrams_dict
    
    def _keyword_extraction(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF-like scoring"""
        if not NLTK_AVAILABLE:
            return []
        
        words = word_tokenize(text.lower())
        words_clean = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
        
        # Calculate term frequency
        word_freq = Counter(words_clean)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Normalize frequencies
        normalized_freq = {
            word: freq / max_freq 
            for word, freq in word_freq.items()
        }
        
        # Sort by frequency
        keywords = sorted(normalized_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [(word, round(score, 3)) for word, score in keywords]
    
    def _detect_technical_terms(self, text: str) -> Dict:
        """Detect domain-specific technical terms"""
        text_lower = text.lower()
        
        technical_categories = {
            'machine_learning': [
                'lstm', 'cnn', 'neural network', 'deep learning', 'random forest',
                'model', 'training', 'prediction', 'classification', 'regression',
                'accuracy', 'precision', 'recall', 'f1-score', 'cross-validation',
                'overfitting', 'underfitting', 'hyperparameter', 'feature engineering',
                'tensorflow', 'pytorch', 'sklearn', 'keras'
            ],
            'data_engineering': [
                'kafka', 'spark', 'hadoop', 'pipeline', 'streaming', 'batch',
                'etl', 'data warehouse', 'data lake', 'postgresql', 'mongodb',
                'redis', 'cassandra', 'elasticsearch', 'airflow', 'dbt'
            ],
            'manufacturing': [
                'bottling', 'sensor', 'equipment', 'maintenance', 'predictive',
                'oee', 'mtbf', 'mttr', 'yield', 'defect', 'quality control',
                'downtime', 'throughput', 'cycle time', 'production line'
            ],
            'api_web': [
                'api', 'rest', 'fastapi', 'flask', 'endpoint', 'http', 'json',
                'request', 'response', 'authentication', 'authorization', 'cors',
                'microservices', 'webhook', 'graphql'
            ],
            'devops': [
                'docker', 'kubernetes', 'container', 'deployment', 'ci/cd',
                'jenkins', 'gitlab', 'monitoring', 'logging', 'prometheus',
                'grafana', 'terraform', 'ansible'
            ]
        }
        
        found_terms = defaultdict(list)
        
        for category, terms in technical_categories.items():
            for term in terms:
                if term in text_lower:
                    # Count occurrences
                    count = text_lower.count(term)
                    found_terms[category].append({'term': term, 'count': count})
        
        # Sort by count within each category
        for category in found_terms:
            found_terms[category] = sorted(found_terms[category], 
                                          key=lambda x: x['count'], 
                                          reverse=True)
        
        return dict(found_terms)
    
    def _document_structure(self, text: str) -> Dict:
        """Analyze document structure"""
        # Detect code blocks
        code_blocks = len(re.findall(r'```[\s\S]*?```|`[^`]+`', text))
        
        # Detect URLs
        urls = re.findall(r'https?://[^\s]+', text)
        
        # Detect markdown headers
        headers = re.findall(r'^#{1,6}\s+.+$', text, re.MULTILINE)
        
        # Detect lists
        bullet_lists = len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
        
        # Detect quotations
        quotes = len(re.findall(r'["\'].*?["\']', text))
        
        # Detect numerical data
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        
        return {
            'code_blocks': code_blocks,
            'urls': len(urls),
            'headers': len(headers),
            'bullet_points': bullet_lists,
            'numbered_lists': numbered_lists,
            'quotations': quotes,
            'numerical_values': len(numbers),
            'has_tables': bool(re.search(r'\|.*\|', text)),
            'has_images': bool(re.search(r'!\[.*?\]\(.*?\)', text))
        }
    
    def print_analysis(self, results: Dict):
        """Pretty print analysis results"""
        print("=" * 80)
        print("COMPREHENSIVE NLP TEXT ANALYSIS")
        print("=" * 80)
        
        # Basic Statistics
        print("\n📊 BASIC STATISTICS")
        print("-" * 80)
        for key, value in results['basic_stats'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Readability
        print("\n📖 READABILITY METRICS")
        print("-" * 80)
        for key, value in results['readability'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Sentiment
        print("\n😊 SENTIMENT ANALYSIS")
        print("-" * 80)
        for method, scores in results['sentiment'].items():
            print(f"\n  {method.upper()}:")
            for key, value in scores.items():
                print(f"    {key.title()}: {value}")
        
        # Keywords
        print("\n🔑 TOP KEYWORDS")
        print("-" * 80)
        for word, score in results['keywords'][:10]:
            print(f"  {word}: {score}")
        
        # Named Entities
        print("\n🏷️  NAMED ENTITIES")
        print("-" * 80)
        for entity_type, entities in results['entities'].items():
            if entities:
                print(f"\n  {entity_type.title()}:")
                for entity in entities[:5]:
                    print(f"    - {entity}")
        
        # Technical Terms
        print("\n🔧 TECHNICAL TERMS")
        print("-" * 80)
        for category, terms in results['technical_terms'].items():
            if terms:
                print(f"\n  {category.replace('_', ' ').title()}:")
                for term_info in terms[:5]:
                    print(f"    - {term_info['term']} (×{term_info['count']})")
        
        # N-grams
        print("\n📝 COMMON PHRASES (N-GRAMS)")
        print("-" * 80)
        for ngram_type, ngrams in results['ngrams'].items():
            print(f"\n  {ngram_type.upper()}:")
            for phrase, count in ngrams[:5]:
                print(f"    {phrase}: {count}")
        
        # Document Structure
        print("\n📄 DOCUMENT STRUCTURE")
        print("-" * 80)
        for key, value in results['document_structure'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "=" * 80)


def main():
    """Example usage"""
    
    # Sample text from your Coca-Cola project
    sample_text = """
    The LSTM-based deep learning model for predictive maintenance achieves 88% accuracy
    in predicting equipment failures 24-48 hours in advance. Using TensorFlow and Keras,
    we implemented a sequential model with dropout regularization to prevent overfitting.
    
    The bottling line sensors collect real-time data including temperature, pressure,
    and vibration measurements. Apache Kafka streams this sensor data to Apache Spark
    for real-time processing and feature engineering. The system maintains an OEE
    (Overall Equipment Effectiveness) of 85% and reduces unplanned downtime by 40%.
    
    FastAPI serves the trained models through RESTful endpoints with sub-100ms latency.
    Docker containers ensure consistent deployment across development and production
    environments. Grafana dashboards provide real-time visualization of KPIs including
    MTBF, MTTR, yield percentage, and defect rates.
    """
    
    # Initialize analyzer
    analyzer = AdvancedNLPAnalyzer()
    
    # Perform analysis
    print("Analyzing text...\n")
    results = analyzer.analyze_text(sample_text)
    
    # Print results
    analyzer.print_analysis(results)
    
    # You can also access specific results
    print("\n\n🎯 QUICK ACCESS EXAMPLES:")
    print(f"Word Count: {results['basic_stats']['word_count']}")
    print(f"Readability: {results['readability']['flesch_interpretation']}")
    print(f"Top Keyword: {results['keywords'][0][0]}")


if __name__ == "__main__":
    main()
