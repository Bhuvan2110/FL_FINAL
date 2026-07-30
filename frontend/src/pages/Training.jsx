


import { useState, useEffect } from 'react'
import { apiFetch } from '../utils/apiFetch'
import LiveChart from '../components/LiveChart'
import {
  Brain, Play, Settings, AlertCircle, Loader2,
  CheckCircle2, Shield, RefreshCw
} from 'lucide-react'

const ALGORITHMS = [
  { value: 'fedavg',   label: 'FedAvg',    desc: 'Weighted average aggregation',          color: 'blue'   },
  { value: 'fedprox',  label: 'FedProx',   desc: 'Proximal term for non-IID stability',   color: 'purple' },
  { value: 'scaffold', label: 'SCAFFOLD',   desc: 'Control variates correct client drift', color: 'green'  },
  { value: 'dpsgd',    label: 'FL + DP-SGD',desc: 'Differential privacy with ε tracking',  color: 'rose'   },
  { value: 'central',  label: 'Central',    desc: 'Centralized baseline (no federation)',  color: 'amber'  },
]

const COLOR_MAP = {
  blue:   { bg: 'bg-blue-500/15',   border: 'border-blue-500/40',   text: 'text-blue-400'    },
  purple: { bg: 'bg-purple-500/15', border: 'border-purple-500/40', text: 'text-purple-400'  },
  green:  { bg: 'bg-emerald-500/15',border: 'border-emerald-500/40',text: 'text-emerald-400' },
  rose:   { bg: 'bg-rose-500/15',   border: 'border-rose-500/40',   text: 'text-rose-400'    },
  amber:  { bg: 'bg-amber-500/15',  border: 'border-amber-500/40',  text: 'text-amber-400'   },
}

export default function Training() {
  const [datasets, setDatasets] = useState([])
  const [experiments, setExperiments] = useState([])
  const [algorithm, setAlgorithm] = useState('fedavg')
  const [config, setConfig] = useState({
    dataset_id: '', n_rounds: 20, lr: 0.01, n_clients: 5, local_epochs: 5,
    mu: 0.1, clip_norm: 1.0, noise_multiplier: 1.0, delta: 1e-5, non_iid: false,
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [activeExp, setActiveExp] = useState(null)

  const loadExperiments = () => {
    apiFetch('/training/list')
      .then(data => setExperiments(Array.isArray(data) ? data : []))
      .catch(() => {})
  }

  useEffect(() => {
    // Load datasets and experiments in parallel, don't block on failure
    apiFetch('/dataset/list').then(data => {
      setDatasets(Array.isArray(data) ? data : [])
      if (data.length > 0) setConfig(c => ({ ...c, dataset_id: data[0].id }))
    }).catch(() => {})
    loadExperiments()
  }, [])

  useEffect(() => {
    const hasActive = experiments.some(e => e.status === 'pending' || e.status === 'running')
    if (!hasActive) return
    const interval = setInterval(loadExperiments, 5000)
    return () => clearInterval(interval)
  }, [experiments])

  const handleStart = async () => {
    if (!config.dataset_id) { setError('Select a dataset'); return }
    setSubmitting(true)
    setError(null)
    try {
      const result = await apiFetch('/training/start', {
        method: 'POST',
        body: JSON.stringify({ algorithm, ...config }),
      })
      setActiveExp({ id: result.experiment_id, algorithm })
      loadExperiments()
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this experiment?')) return
    // Optimistic removal
    setExperiments(prev => prev.filter(e => e.id !== id))
    if (activeExp && activeExp.id === id) {
      setActiveExp(null)
    }
    try {
      await apiFetch(`/training/${id}`, { method: 'DELETE' })
    } catch (e) {
      alert(e.message)
      loadExperiments() // Restore on failure
    }
  }

  const algoInfo = ALGORITHMS.find(a => a.value === algorithm)
  const colrs = COLOR_MAP[algoInfo?.color || 'blue']

  return (
    <div className="page-wrapper">
      <h1 className="page-title">Training</h1>
      <p className="page-subtitle">Configure and launch federated learning experiments</p>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Config panel */}
        <div className="space-y-4">
          {/* Algorithm selector */}
          <div className="glass-card p-5">
            <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Brain size={14} className="text-brand-400" /> Algorithm
            </h2>
            <div className="grid grid-cols-1 gap-2">
              {ALGORITHMS.map(algo => {
                const c = COLOR_MAP[algo.color]
                const selected = algorithm === algo.value
                return (
                  <button
                    key={algo.value}
                    onClick={() => setAlgorithm(algo.value)}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all duration-200
                      ${selected ? `${c.bg} ${c.border}` : 'border-white/5 hover:border-white/15 hover:bg-white/3'}`}
                  >
                    <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${selected ? `${c.text.replace('text-', 'bg-')}` : 'bg-slate-600'}`} />
                    <div>
                      <p className={`text-sm font-semibold ${selected ? c.text : 'text-slate-300'}`}>{algo.label}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{algo.desc}</p>
                    </div>
                    {algo.value === 'dpsgd' && (
                      <Shield size={12} className="text-rose-400 ml-auto mt-0.5 flex-shrink-0" />
                    )}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="glass-card p-5">
            <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Settings size={14} className="text-brand-400" /> Hyperparameters
            </h2>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Dataset</label>
                <select
                  value={config.dataset_id}
                  onChange={e => setConfig(c => ({ ...c, dataset_id: e.target.value }))}
                  className="input-field"
                >
                  <option value="">— Select dataset —</option>
                  {datasets.map(d => <option key={d.id} value={d.id}>{d.filename} ({d.row_count?.toLocaleString()} rows)</option>)}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { key: 'n_rounds',      label: 'Rounds',      type: 'number', min: 1, max: 100 },
                  { key: 'lr',            label: 'Learning Rate', type: 'number', step: 0.001,  min: 0.0001, max: 1 },
                  { key: 'n_clients',     label: 'Clients',     type: 'number', min: 2, max: 50, hide: algorithm === 'central' },
                  { key: 'local_epochs',  label: 'Local Epochs', type: 'number', min: 1, max: 20 },
                ].filter(f => !f.hide).map(field => (
                  <div key={field.key}>
                    <label className="text-xs text-slate-400 mb-1 block">{field.label}</label>
                    <input
                      type={field.type}
                      step={field.step}
                      min={field.min}
                      max={field.max}
                      value={config[field.key]}
                      onChange={e => setConfig(c => ({ ...c, [field.key]: parseFloat(e.target.value) || e.target.value }))}
                      className="input-field"
                    />
                  </div>
                ))}
              </div>

              {/* Non-IID toggle */}
              {algorithm !== 'central' && (
                <label className="flex items-center gap-3 cursor-pointer">
                  <div className="relative">
                    <input type="checkbox" className="sr-only" checked={config.non_iid} onChange={e => setConfig(c => ({ ...c, non_iid: e.target.checked }))} />
                    <div className={`w-10 h-5 rounded-full transition-colors ${config.non_iid ? 'bg-brand-600' : 'bg-surface-600'}`} />
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${config.non_iid ? 'translate-x-5' : ''}`} />
                  </div>
                  <span className="text-xs text-slate-300">Non-IID data partition</span>
                </label>
              )}

              {/* FedProx */}
              {algorithm === 'fedprox' && (
                <div>
                  <label className="text-xs text-slate-400 mb-1 block">Proximal Term μ</label>
                  <input type="number" step={0.01} min={0} value={config.mu} onChange={e => setConfig(c => ({ ...c, mu: parseFloat(e.target.value) }))} className="input-field" />
                </div>
              )}

              {/* DP-SGD */}
              {algorithm === 'dpsgd' && (
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { key: 'clip_norm',        label: 'Clip Norm C',       step: 0.1  },
                    { key: 'noise_multiplier', label: 'Noise Multiplier σ', step: 0.1  },
                  ].map(f => (
                    <div key={f.key}>
                      <label className="text-xs text-slate-400 mb-1 block">{f.label}</label>
                      <input type="number" step={f.step} min={0} value={config[f.key]} onChange={e => setConfig(c => ({ ...c, [f.key]: parseFloat(e.target.value) }))} className="input-field" />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {error && (
              <div className="mt-3 flex items-center gap-2 text-red-400 text-xs p-3 rounded-xl bg-red-500/10 border border-red-500/30">
                <AlertCircle size={13} />
                {error}
              </div>
            )}

            <button
              onClick={handleStart}
              disabled={submitting || !config.dataset_id}
              className={`w-full mt-4 py-3 rounded-xl font-semibold text-sm transition-all duration-200 active:scale-98
                          flex items-center justify-center gap-2 shadow-lg disabled:opacity-50
                          ${colrs.bg} ${colrs.text} border ${colrs.border} hover:brightness-110`}
            >
              {submitting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
              {submitting ? 'Launching…' : `Start ${algoInfo?.label}`}
            </button>
          </div>
        </div>

        {/* Live chart */}
        <div className="space-y-4">
          {activeExp ? (
            <LiveChart experimentId={activeExp.id} algorithm={activeExp.algorithm} />
          ) : (
            <div className="glass-card p-12 text-center">
              <Brain size={40} className="text-slate-700 mx-auto mb-4" />
              <p className="text-slate-500 text-sm">Configure and start training to see the live chart</p>
            </div>
          )}

          {activeExp && (
            <div className="glass-card p-4 flex items-center gap-3">
              <CheckCircle2 size={15} className="text-emerald-400" />
              <div>
                <p className="text-xs font-medium text-slate-200">Experiment started</p>
                <p className="text-xs text-slate-500 font-mono">{activeExp.id}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Experiments / Trained Results */}
      <div className="glass-card mt-6">
        <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200">Recent Experiments</h2>
          <button onClick={loadExperiments} className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date/Time</th>
                <th>Algorithm</th>
                <th>Dataset</th>
                <th>Rounds</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-6 text-slate-500 text-sm">
                    No experiments found. Start a training session above!
                  </td>
                </tr>
              ) : (
                experiments.map(exp => (
                  <tr key={exp.id}>
                    <td className="text-slate-400 text-xs font-mono">
                      {new Date(exp.created_at).toLocaleString()}
                    </td>
                    <td>
                      <span className={`badge-${exp.algorithm}`}>
                        {ALGORITHMS.find(a => a.value === exp.algorithm)?.label || exp.algorithm}
                      </span>
                    </td>
                    <td className="text-slate-300 text-sm">
                      {exp.datasets?.filename || 'Unknown'}
                    </td>
                    <td className="text-slate-400 font-mono text-xs">
                      {exp.hyperparams?.n_rounds || '—'}
                    </td>
                    <td>
                      <span className={`badge-${exp.status}`}>
                        {exp.status}
                      </span>
                    </td>
                    <td>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setActiveExp({ id: exp.id, algorithm: exp.algorithm })}
                          className="px-2 py-1 rounded bg-brand-500/10 hover:bg-brand-500/20 text-brand-400 text-xs font-medium transition-colors"
                        >
                          View Progress
                        </button>
                        <button
                          onClick={() => handleDelete(exp.id)}
                          className="px-2 py-1 rounded bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 text-xs font-medium transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
