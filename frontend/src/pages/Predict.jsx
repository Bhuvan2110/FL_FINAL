import { useState, useEffect } from 'react'
import { apiFetch, apiUpload } from '../utils/apiFetch'
import { Zap, Upload, Download, AlertCircle, Loader2, CheckCircle2 } from 'lucide-react'

export default function Predict() {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [features, setFeatures] = useState({})
  const [result, setResult] = useState(null)
  const [batchResults, setBatchResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('single')

  useEffect(() => {
    // Load models in background — don't block the page
    apiFetch('/predict/models').then(data => {
      setModels(Array.isArray(data) ? data : [])
    }).catch(() => {})
  }, [])

  const handleSinglePredict = async () => {
    if (!selectedModel) { setError('Select a model'); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await apiFetch('/predict/single', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel.id, features }),
      })
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleBatchPredict = async (file) => {
    if (!selectedModel) { setError('Select a model'); return }
    setLoading(true)
    setError(null)
    setBatchResults(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await apiUpload(`/predict/batch?model_id=${selectedModel.id}`, form)
      setBatchResults(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const exportBatch = () => {
    if (!batchResults) return
    const rows = batchResults.results.map(r =>
      `${r.row},${r.prediction},${r.confidence},${r.class_label}`
    )
    const csv = ['row,prediction,confidence,class_label', ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
  }

  const ALGO_COLORS = {
    fedavg: 'badge-fedavg', fedprox: 'badge-fedprox', scaffold: 'badge-scaffold',
    dpsgd: 'badge-dpsgd', central: 'badge-central',
  }

  return (
    <div className="page-wrapper">
      <h1 className="page-title">Predict</h1>
      <p className="page-subtitle">Run single or batch predictions using trained FL models</p>

      {/* Model selector */}
      <div className="glass-card p-5 mb-5">
        <h2 className="text-sm font-semibold text-slate-200 mb-3">Select Model</h2>
        {models.length === 0 ? (
          <p className="text-slate-500 text-sm">No trained models yet — complete at least one training run first.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {models.map(m => (
              <button
                key={m.id}
                onClick={() => setSelectedModel(m)}
                className={`flex items-center gap-3 p-3 rounded-xl border text-left transition-all
                  ${selectedModel?.id === m.id ? 'border-brand-500/50 bg-brand-500/10' : 'border-white/5 hover:border-white/15'}`}
              >
                <Zap size={14} className="text-brand-400 flex-shrink-0" />
                <div className="min-w-0">
                  <span className={ALGO_COLORS[m.algorithm] || ''}>{m.algorithm}</span>
                  <p className="text-xs text-slate-500 font-mono mt-0.5 truncate">v{m.version} · {m.id.slice(0, 8)}…</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2 mb-5 p-1 glass-card w-fit rounded-xl">
        {['single', 'batch'].map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize
              ${mode === m ? 'bg-brand-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}>
            {m === 'single' ? 'Single Prediction' : 'Batch CSV'}
          </button>
        ))}
      </div>

      {mode === 'single' ? (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-200 mb-4">Feature Values</h2>
          {selectedModel ? (
            <div className="space-y-3">
              <p className="text-xs text-slate-400">Enter feature values as JSON key-value pairs</p>
              <textarea
                rows={6}
                className="input-field font-mono text-xs resize-none"
                placeholder='{"age": 45, "glucose": 120, "bmi": 28.5, ...}'
                onChange={e => {
                  try { setFeatures(JSON.parse(e.target.value)) } catch { /* invalid JSON ignored until valid */ }
                }}
              />
              <button onClick={handleSinglePredict} disabled={loading} className="btn-primary">
                {loading ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
                {loading ? 'Predicting…' : 'Predict'}
              </button>

              {result && (
                <div className={`p-4 rounded-xl border animate-slide-up
                  ${result.prediction === 1 ? 'bg-rose-500/10 border-rose-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
                  <div className="flex items-center gap-3">
                    <CheckCircle2 size={20} className={result.prediction === 1 ? 'text-rose-400' : 'text-emerald-400'} />
                    <div>
                      <p className="font-bold text-lg text-slate-100">{result.class_label}</p>
                      <p className="text-xs text-slate-400">
                        Confidence: <span className="font-mono text-slate-200">{(result.confidence * 100).toFixed(1)}%</span>
                        <span className="mx-2">·</span>
                        Hash: <span className="font-mono">{result.input_hash}</span>
                      </p>
                    </div>
                  </div>
                  {/* Confidence bar */}
                  <div className="mt-3 h-2 rounded-full bg-surface-700 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${result.prediction === 1 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-slate-500 text-sm">Select a model above to enter features.</p>
          )}
        </div>
      ) : (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-200 mb-4">Batch Prediction</h2>
          <div
            className="drop-zone mb-4"
            onClick={() => document.getElementById('batch-csv').click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); handleBatchPredict(e.dataTransfer.files[0]) }}
          >
            <input id="batch-csv" type="file" accept=".csv" className="hidden" onChange={e => handleBatchPredict(e.target.files[0])} />
            {loading ? (
              <Loader2 size={24} className="text-brand-400 animate-spin mx-auto" />
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload size={24} className="text-brand-400" />
                <p className="text-slate-400 text-sm">Drop batch CSV or click to upload</p>
              </div>
            )}
          </div>

          {batchResults && (
            <div className="animate-slide-up">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm text-slate-200">{batchResults.total} rows processed</p>
                <button onClick={exportBatch} className="btn-secondary text-xs">
                  <Download size={12} /> Export CSV
                </button>
              </div>
              <div className="overflow-x-auto max-h-64">
                <table className="data-table text-xs">
                  <thead>
                    <tr><th>Row</th><th>Prediction</th><th>Confidence</th><th>Class</th></tr>
                  </thead>
                  <tbody>
                    {batchResults.results.slice(0, 50).map(r => (
                      <tr key={r.row}>
                        <td className="font-mono">{r.row}</td>
                        <td className="font-mono">{r.prediction ?? '—'}</td>
                        <td className="font-mono">{r.confidence !== undefined ? `${(r.confidence * 100).toFixed(1)}%` : '—'}</td>
                        <td>
                          {r.class_label && (
                            <span className={`text-xs font-medium ${r.class_label === 'Positive' ? 'text-rose-400' : 'text-emerald-400'}`}>
                              {r.class_label}
                            </span>
                          )}
                          {r.error && <span className="text-red-400 text-xs">{r.error}</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 glass-card p-4 flex items-center gap-3 text-red-400 border border-red-500/30">
          <AlertCircle size={15} />
          <span className="text-sm">{error}</span>
        </div>
      )}
    </div>
  )
}
