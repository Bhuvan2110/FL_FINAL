import { useState, useEffect, useRef } from 'react'
import { apiFetch } from '../utils/apiFetch'
import { Send, Bot, User, RefreshCw, AlertCircle, Loader2, Sparkles, Sliders } from 'lucide-react'

export default function AgentChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your FL Predictive Analytics AI Agent. 🤖\n\nI can guide you through running predictions using our trained Federated Learning models. Please select a model below to begin.'
    }
  ])
  const [inputText, setInputText] = useState('')
  const [selectedModel, setSelectedModel] = useState(null)
  const [models, setModels] = useState([])
  const [currentFeature, setCurrentFeature] = useState(null)
  const [collectedFeatures, setCollectedFeatures] = useState({})
  const [options, setOptions] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const chatEndRef = useRef(null)

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, options, loading])

  // Load models initially
  useEffect(() => {
    apiFetch('/predict/models')
      .then(data => {
        const list = Array.isArray(data) ? data : []
        setModels(list)
        if (list.length > 0 && !selectedModel) {
          setOptions(list.map(m => ({
            value: m.id,
            label: `${m.algorithm.toUpperCase()} (v${m.version} · ${m.id.slice(0, 8)})`
          })))
        } else if (list.length === 0) {
          setMessages([
            {
              role: 'assistant',
              content: 'Welcome! I am your AI Chat Agent. 🤖\n\nIt looks like there are no trained models available yet. Please run an FL training task first in the Training section!'
            }
          ])
        }
      })
      .catch(err => {
        setError('Failed to load models: ' + err.message)
      })
  }, [selectedModel])

  const sendMessage = async (text, val = null) => {
    if (!text && !val) return

    const userText = text || (val ? `Selected: ${val.label || val.value || val}` : '')
    const userValue = val ? (val.value !== undefined ? val.value : val) : text

    // 1. Add user message
    const updatedMessages = [...messages, { role: 'user', content: userText }]
    setMessages(updatedMessages)
    setInputText('')
    setLoading(true)
    setError(null)

    // Handle Model Selection
    let modelId = selectedModel?.id || null
    if (!selectedModel && val) {
      const selected = models.find(m => m.id === userValue)
      if (selected) {
        setSelectedModel(selected)
        modelId = selected.id
      }
    }

    try {
      // 2. Post to AI Agent chat endpoint
      const res = await apiFetch('/agent/chat', {
        method: 'POST',
        body: JSON.stringify({
          model_id: modelId,
          messages: updatedMessages,
          current_feature: currentFeature,
          collected_features: collectedFeatures
        })
      })

      // 3. Update state with agent response
      setMessages(prev => [...prev, { role: 'assistant', content: res.message }])
      setCurrentFeature(res.current_feature)
      setCollectedFeatures(res.collected_features)
      setOptions(res.options || [])
      
      if (res.prediction) {
        setPrediction(res.prediction)
      } else {
        // If restarting or switching
        if (userValue === 'restart') {
          setPrediction(null)
          setCollectedFeatures({})
          setCurrentFeature(null)
        } else if (userValue === 'switch_model') {
          setSelectedModel(null)
          setPrediction(null)
          setCollectedFeatures({})
          setCurrentFeature(null)
          setOptions(models.map(m => ({
            value: m.id,
            label: `${m.algorithm.toUpperCase()} (v${m.version} · ${m.id.slice(0, 8)})`
          })))
        }
      }
    } catch (err) {
      setError(err.message)
      // Pop the user message if failed to keep conversation clean, or just show error
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedModel(null)
    setPrediction(null)
    setCollectedFeatures({})
    setCurrentFeature(null)
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! Chat restarted. Please select a model below to begin.'
      }
    ])
    if (models.length > 0) {
      setOptions(models.map(m => ({
        value: m.id,
        label: `${m.algorithm.toUpperCase()} (v${m.version} · ${m.id.slice(0, 8)})`
      })))
    } else {
      setOptions([])
    }
    setError(null)
  }

  return (
    <div className="page-wrapper flex flex-col h-[calc(100vh-1rem)]">
      <div className="flex items-center justify-between border-b border-white/5 pb-4 mb-4">
        <div>
          <h1 className="page-title mb-0 flex items-center gap-2">
            <Sparkles className="text-brand-400" size={24} /> AI Chat Agent
          </h1>
          <p className="page-subtitle mb-0">Talk to the AI clinician agent to gather feature inputs and run predictions</p>
        </div>
        <button onClick={handleReset} className="btn-secondary text-xs flex items-center gap-1.5 py-2">
          <RefreshCw size={12} /> Reset Chat
        </button>
      </div>

      <div className="flex-1 flex gap-5 min-h-0">
        {/* Chat box */}
        <div className="flex-1 flex flex-col glass-card p-0 overflow-hidden relative border border-white/10 rounded-2xl bg-surface-900/60">
          {/* Header */}
          <div className="px-5 py-3.5 bg-surface-800/50 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-semibold text-slate-300">AI Assistant Online</span>
            </div>
            {selectedModel && (
              <div className="text-xs text-brand-400 font-medium px-2 py-0.5 rounded-full bg-brand-500/10 border border-brand-500/20">
                Model: {selectedModel.algorithm.toUpperCase()} (v{selectedModel.version})
              </div>
            )}
          </div>

          {/* Messages scrollarea */}
          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
            {messages.map((m, idx) => (
              <div key={idx} className={`flex gap-3 max-w-[85%] ${m.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center border shadow-sm
                  ${m.role === 'user' 
                    ? 'bg-gradient-to-br from-brand-600 to-purple-600 border-brand-400/20 text-white' 
                    : 'bg-surface-800 border-white/15 text-slate-300'}`}>
                  {m.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                </div>

                <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap transition-all duration-300
                  ${m.role === 'user'
                    ? 'bg-brand-600/90 text-white rounded-tr-none shadow-md shadow-brand-900/20'
                    : 'bg-surface-800/80 border border-white/5 text-slate-100 rounded-tl-none'}`}>
                  {m.content}
                </div>
              </div>
            ))}

            {/* Options selection pills */}
            {options.length > 0 && !loading && (
              <div className="flex flex-wrap gap-2 pt-2 justify-center max-w-[90%] mx-auto bg-surface-900/20 p-3 rounded-xl border border-white/5">
                {options.map((opt, oIdx) => (
                  <button
                    key={oIdx}
                    onClick={() => sendMessage(null, opt)}
                    className="px-3.5 py-1.5 text-xs font-semibold rounded-full border border-brand-500/30 bg-brand-500/5 hover:bg-brand-500/20 text-brand-300 transition-all hover:scale-[1.03] active:scale-[0.98]"
                  >
                    {opt.label || opt.value || opt}
                  </button>
                ))}
              </div>
            )}

            {/* Typing Loader */}
            {loading && (
              <div className="flex gap-3 max-w-[80%]">
                <div className="w-8 h-8 rounded-full bg-surface-800 border border-white/15 text-slate-300 flex items-center justify-center">
                  <Bot size={14} />
                </div>
                <div className="bg-surface-800/50 border border-white/5 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-1">
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Error Message */}
          {error && (
            <div className="mx-5 mb-3 p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl flex items-center gap-2.5 text-rose-400 text-xs">
              <AlertCircle size={14} className="flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Form input */}
          <div className="p-4 bg-surface-800/30 border-t border-white/10 flex gap-3">
            <input
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') sendMessage(inputText)
              }}
              placeholder={
                currentFeature 
                  ? `Enter value for ${currentFeature}...` 
                  : selectedModel 
                    ? "Chat with the agent..." 
                    : "Please select a model above..."
              }
              className="flex-1 input-field"
              disabled={loading || options.length > 0 && !selectedModel}
            />
            <button
              onClick={() => sendMessage(inputText)}
              disabled={loading || !inputText.trim() || (options.length > 0 && !selectedModel)}
              className="btn-primary px-4"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>

        {/* Info panel / Features checklist */}
        <div className="w-80 glass-card flex flex-col p-4 border border-white/10 rounded-2xl bg-surface-900/60 overflow-y-auto">
          {selectedModel ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-1.5">
                  <Sliders size={15} className="text-brand-400" /> Feature Checklist
                </h3>
                <p className="text-xs text-slate-500">Collects inputs used to train {selectedModel.algorithm.toUpperCase()}</p>
              </div>

              <div className="space-y-2">
                {selectedModel.feature_names?.map(f => {
                  const hasVal = collectedFeatures[f] !== undefined
                  const isCurrent = currentFeature === f
                  return (
                    <div
                      key={f}
                      className={`flex items-center justify-between p-2 rounded-xl border text-xs transition-all duration-200
                        ${hasVal
                          ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                          : isCurrent
                            ? 'bg-brand-500/10 border-brand-500/30 text-brand-300 font-medium scale-[1.02]'
                            : 'bg-white/5 border-white/5 text-slate-400'}`}
                    >
                      <span className="font-mono font-medium truncate max-w-[150px]">{f}</span>
                      <span className="font-mono">
                        {hasVal ? (
                          <span className="bg-emerald-500/20 px-2 py-0.5 rounded-md text-emerald-300">
                            {collectedFeatures[f].toString().slice(0, 10)}
                          </span>
                        ) : isCurrent ? (
                          <span className="animate-pulse bg-brand-500/20 text-brand-400 px-2 py-0.5 rounded-md">asking...</span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </span>
                    </div>
                  )
                })}
              </div>

              {/* Prediction details */}
              {prediction && (
                <div className={`p-4 rounded-xl border animate-slide-up mt-4
                  ${prediction.prediction === 1 ? 'bg-rose-500/10 border-rose-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
                  <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-2">Prediction Outcome</h4>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-xl font-bold ${prediction.prediction === 1 ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {prediction.class_label}
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      ({(prediction.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 w-full bg-surface-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${prediction.prediction === 1 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-4">
              <Bot size={36} className="text-slate-600 mb-3 animate-pulse" />
              <p className="text-sm text-slate-400 font-semibold mb-1">Select a Model first</p>
              <p className="text-xs text-slate-500 leading-relaxed">
                Choose one of our Federated Learning models in the chat box to begin the step-by-step feature collection.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
