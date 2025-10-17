import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { FileText, Search, TrendingUp, Hash, Clock, BookOpen } from 'lucide-react';

const TextAnalyzer = () => {
  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState(null);

  const analyzeText = () => {
    if (!text.trim()) return;

    // Basic statistics
    const words = text.trim().split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const characters = text.length;
    const charactersNoSpaces = text.replace(/\s/g, '').length;
    
    // Word frequency
    const wordFreq = {};
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'its']);
    
    words.forEach(word => {
      const cleaned = word.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (cleaned && cleaned.length > 2 && !stopWords.has(cleaned)) {
        wordFreq[cleaned] = (wordFreq[cleaned] || 0) + 1;
      }
    });

    const topWords = Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word, count]) => ({ word, count }));

    // Technical terms detection
    const technicalTerms = {
      'ml': ['machine learning', 'lstm', 'cnn', 'model', 'training', 'prediction', 'accuracy', 'neural', 'tensorflow', 'sklearn'],
      'data': ['data', 'pipeline', 'kafka', 'spark', 'streaming', 'database', 'postgresql', 'redis'],
      'production': ['bottling', 'sensor', 'equipment', 'maintenance', 'quality', 'defect', 'oee', 'yield'],
      'api': ['api', 'fastapi', 'endpoint', 'request', 'response', 'http', 'rest']
    };

    const foundTerms = {};
    Object.entries(technicalTerms).forEach(([category, terms]) => {
      const found = terms.filter(term => 
        text.toLowerCase().includes(term)
      );
      if (found.length > 0) {
        foundTerms[category] = found;
      }
    });

    // Sentence complexity
    const avgWordsPerSentence = words.length / sentences.length;
    const avgCharsPerWord = charactersNoSpaces / words.length;

    // Readability (simplified Flesch Reading Ease)
    const syllables = words.reduce((sum, word) => sum + estimateSyllables(word), 0);
    const fleschScore = 206.835 - 1.015 * (words.length / sentences.length) - 84.6 * (syllables / words.length);
    
    let readabilityLevel;
    if (fleschScore >= 90) readabilityLevel = 'Very Easy';
    else if (fleschScore >= 80) readabilityLevel = 'Easy';
    else if (fleschScore >= 70) readabilityLevel = 'Fairly Easy';
    else if (fleschScore >= 60) readabilityLevel = 'Standard';
    else if (fleschScore >= 50) readabilityLevel = 'Fairly Difficult';
    else if (fleschScore >= 30) readabilityLevel = 'Difficult';
    else readabilityLevel = 'Very Difficult';

    // Sentiment analysis (simplified)
    const positiveWords = ['good', 'great', 'excellent', 'improve', 'success', 'efficient', 'optimize', 'better', 'increase', 'achieve'];
    const negativeWords = ['bad', 'poor', 'fail', 'error', 'problem', 'issue', 'decrease', 'reduce', 'difficult', 'complex'];
    
    const lowerText = text.toLowerCase();
    const positiveCount = positiveWords.filter(w => lowerText.includes(w)).length;
    const negativeCount = negativeWords.filter(w => lowerText.includes(w)).length;
    
    let sentiment = 'Neutral';
    if (positiveCount > negativeCount + 1) sentiment = 'Positive';
    else if (negativeCount > positiveCount + 1) sentiment = 'Negative';

    // Code detection
    const hasCode = /```|def |class |import |function|const |let |var /.test(text);
    const hasNumbers = /\d+/.test(text);
    const hasUrls = /https?:\/\//.test(text);

    // Unique words
    const uniqueWords = new Set(words.map(w => w.toLowerCase().replace(/[^a-z0-9]/g, '')));

    // Reading time (avg 200 words per minute)
    const readingTime = Math.ceil(words.length / 200);

    setAnalysis({
      basic: {
        characters,
        charactersNoSpaces,
        words: words.length,
        sentences: sentences.length,
        paragraphs: text.split(/\n\n+/).length,
        uniqueWords: uniqueWords.size
      },
      advanced: {
        avgWordsPerSentence: avgWordsPerSentence.toFixed(1),
        avgCharsPerWord: avgCharsPerWord.toFixed(1),
        lexicalDiversity: (uniqueWords.size / words.length * 100).toFixed(1),
        readabilityLevel,
        fleschScore: fleschScore.toFixed(1),
        sentiment,
        readingTime
      },
      topWords,
      technicalTerms: foundTerms,
      features: {
        hasCode,
        hasNumbers,
        hasUrls
      }
    });
  };

  const estimateSyllables = (word) => {
    word = word.toLowerCase().replace(/[^a-z]/g, '');
    if (word.length <= 3) return 1;
    const vowels = word.match(/[aeiouy]+/g);
    return vowels ? vowels.length : 1;
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-8 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <FileText className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">Advanced Text Paragraph Analyzer</h1>
          </div>
          
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Paste your paragraph here:
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter or paste any paragraph from your documents for comprehensive analysis..."
              className="w-full h-48 p-4 border-2 border-gray-300 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all resize-none"
            />
          </div>

          <button
            onClick={analyzeText}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Search className="w-5 h-5" />
            Analyze Text
          </button>
        </div>

        {analysis && (
          <div className="space-y-6">
            {/* Basic Statistics */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Hash className="w-6 h-6 text-indigo-600" />
                Basic Statistics
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {Object.entries(analysis.basic).map(([key, value]) => (
                  <div key={key} className="bg-gradient-to-br from-indigo-50 to-blue-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-indigo-600">{value}</div>
                    <div className="text-sm text-gray-600 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Advanced Metrics */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-indigo-600" />
                Advanced Metrics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
                  <div className="text-sm text-gray-600 mb-1">Readability Level</div>
                  <div className="text-xl font-bold text-green-700">{analysis.advanced.readabilityLevel}</div>
                  <div className="text-xs text-gray-500">Flesch Score: {analysis.advanced.fleschScore}</div>
                </div>
                
                <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
                  <div className="text-sm text-gray-600 mb-1">Avg Words/Sentence</div>
                  <div className="text-xl font-bold text-blue-700">{analysis.advanced.avgWordsPerSentence}</div>
                </div>
                
                <div className="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-500">
                  <div className="text-sm text-gray-600 mb-1">Lexical Diversity</div>
                  <div className="text-xl font-bold text-purple-700">{analysis.advanced.lexicalDiversity}%</div>
                </div>
                
                <div className="bg-amber-50 p-4 rounded-lg border-l-4 border-amber-500">
                  <div className="text-sm text-gray-600 mb-1">Sentiment</div>
                  <div className="text-xl font-bold text-amber-700">{analysis.advanced.sentiment}</div>
                </div>
              </div>

              <div className="mt-4 flex items-center gap-4 bg-gray-50 p-4 rounded-lg">
                <Clock className="w-6 h-6 text-gray-600" />
                <div>
                  <span className="font-semibold text-gray-700">Estimated Reading Time: </span>
                  <span className="text-gray-600">{analysis.advanced.readingTime} minute{analysis.advanced.readingTime !== 1 ? 's' : ''}</span>
                </div>
              </div>
            </div>

            {/* Word Frequency Chart */}
            {analysis.topWords.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Top 10 Most Frequent Words</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analysis.topWords}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="word" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#6366f1" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Technical Terms */}
            {Object.keys(analysis.technicalTerms).length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <BookOpen className="w-6 h-6 text-indigo-600" />
                  Technical Terms Detected
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(analysis.technicalTerms).map(([category, terms]) => (
                    <div key={category} className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="font-semibold text-gray-700 mb-2 capitalize">{category}</h3>
                      <div className="flex flex-wrap gap-2">
                        {terms.map((term, idx) => (
                          <span key={idx} className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full text-sm">
                            {term}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Content Features */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Content Features</h2>
              <div className="flex flex-wrap gap-4">
                {analysis.features.hasCode && (
                  <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg font-semibold">
                    ✓ Contains Code
                  </div>
                )}
                {analysis.features.hasNumbers && (
                  <div className="bg-blue-100 text-blue-800 px-4 py-2 rounded-lg font-semibold">
                    ✓ Contains Numbers
                  </div>
                )}
                {analysis.features.hasUrls && (
                  <div className="bg-purple-100 text-purple-800 px-4 py-2 rounded-lg font-semibold">
                    ✓ Contains URLs
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextAnalyzer;
