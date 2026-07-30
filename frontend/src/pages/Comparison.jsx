import { useState, useEffect } from 'react'
import { apiFetch } from '../utils/apiFetch'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { BarChart3, Download, RefreshCw, AlertCircle, TrendingUp, Shield, Target } from 'lucide-react'

const ALGO_META = {
  fedavg:   { color: '#3b82f6', label: 'FedAvg'   },
  fedprox:  { color: '#8b5cf6', label: 'FedProx'  },
  scaffold: { color: '#10b981', label: 'SCAFFOLD'  },
  dpsgd:    { color: '#f43f5e', label: 'FL+DP-SGD' },
  central:  { color: '#f59e0b', label: 'Central'   },
}

const getExpLabel = (c) => {
  const shortId = c.experiment_id ? c.experiment_id.slice(0, 4) : ''
  return `${ALGO_META[c.algorithm]?.label || c.algorithm} (${shortId})`
}

function MetricBar({ label, compare }) {
  const data = compare.map(c => ({
    name: getExpLabel(c),
    value: c.metrics?.[label] || 0,
    color: ALGO_META[c.algorithm]?.color || '#3b82f6',
  }))

  return (
    <div className="glass-card p-4">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">{label.toUpperCase()}</p>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data} margin={{ top: 0, right: 10, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
          <Tooltip
            formatter={(v) => [(v * 100).toFixed(1) + '%', label]}
            contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => <Cell key={i} fill={entry.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ConfusionMatrix({ cm, algorithm }) {
  if (!cm) return null
  const { tp, tn, fp, fn } = cm
  const cells = [
    { label: 'TN', value: tn, bg: 'bg-emerald-500/30', text: 'text-emerald-300' },
    { label: 'FP', value: fp, bg: 'bg-rose-500/20',    text: 'text-rose-300'    },
    { label: 'FN', value: fn, bg: 'bg-rose-500/20',    text: 'text-rose-300'    },
    { label: 'TP', value: tp, bg: 'bg-emerald-500/30', text: 'text-emerald-300' },
  ]
  return (
    <div>
      <p className="text-xs text-slate-500 mb-2">Confusion Matrix — {algorithm}</p>
      <div className="grid grid-cols-2 gap-1.5 w-40">
        {cells.map(c => (
          <div key={c.label} className={`heatmap-cell h-14 flex-col gap-0.5 ${c.bg}`}>
            <span className={`text-lg ${c.text}`}>{c.value}</span>
            <span className="text-xs text-slate-500">{c.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConvergenceChart({ compare }) {
  const allRounds = new Set(compare.flatMap(c => (c.rounds || []).map(r => r.round_num)))
  const maxRound = Math.max(...allRounds, 0)
  const data = Array.from({ length: maxRound }, (_, i) => {
    const row = { round: i + 1 }
    compare.forEach(c => {
      const rd = (c.rounds || []).find(r => r.round_num === i + 1)
      if (rd) row[getExpLabel(c)] = rd.val_accuracy ?? rd.accuracy
    })
    return row
  })

  return (
    <div className="glass-card p-5">
      <p className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <TrendingUp size={14} className="text-brand-400" /> Convergence Curves
      </p>
      {data.length === 0 ? (
        <p className="text-slate-500 text-sm text-center py-8">No round data available</p>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 0, right: 15, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="round" tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            {compare.map(c => (
              <Line key={c.experiment_id} type="monotone" dataKey={getExpLabel(c)}
                stroke={ALGO_META[c.algorithm]?.color || '#3b82f6'} strokeWidth={2} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

function FeatureImportanceChart({ compare }) {
  const first = compare.find(c => c.metrics?.feature_importance?.length > 0)
  if (!first) return null
  const fi = (first.metrics.feature_importance || []).slice(0, 10)
  return (
    <div className="glass-card p-5">
      <p className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Target size={14} className="text-brand-400" /> Feature Importance ({getExpLabel(first)})
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart layout="vertical" data={fi} margin={{ top: 0, right: 20, bottom: 0, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
          <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: '#94a3b8' }} width={60} />
          <Tooltip
            formatter={(v) => [(v * 100).toFixed(1) + '%', 'Importance']}
            contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
          />
          <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function PrivacyUtilityChart({ privacyCurve }) {
  if (!privacyCurve?.length) return null
  return (
    <div className="glass-card p-5">
      <p className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Shield size={14} className="text-rose-400" /> Privacy-Utility Tradeoff
        <span className="text-xs text-slate-500 ml-1">(accuracy vs ε)</span>
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={privacyCurve} margin={{ top: 0, right: 15, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="epsilon" label={{ value: 'ε (epsilon)', position: 'insideBottom', fontSize: 10, fill: '#64748b' }} tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} />
          <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }} />
          <Line type="monotone" dataKey="accuracy" stroke="#f43f5e" strokeWidth={2} dot={{ r: 3, fill: '#f43f5e' }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Comparison() {
  const [compare, setCompare] = useState([])
  const [selectedExpIds, setSelectedExpIds] = useState([])
  const [privacyCurve, setPrivacyCurve] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [cmpResult, pucResult] = await Promise.allSettled([
        apiFetch('/training/compare'),
        apiFetch('/metrics/privacy-utility'),
      ])
      const cmpData = cmpResult.status === 'fulfilled' && Array.isArray(cmpResult.value) ? cmpResult.value : []
      setCompare(cmpData)
      setPrivacyCurve(pucResult.status === 'fulfilled' && Array.isArray(pucResult.value) ? pucResult.value : [])
      setSelectedExpIds(cmpData.map(c => c.experiment_id))
      if (cmpResult.status === 'rejected') {
        setError(cmpResult.reason?.message || 'Failed to load comparison data')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const exportMetrics = () => {
    const activeCompare = compare.filter(c => selectedExpIds.includes(c.experiment_id))
    const data = activeCompare.map(c => ({
      algorithm: c.algorithm,
      accuracy: c.metrics?.accuracy,
      f1: c.metrics?.f1,
      auc: c.metrics?.auc,
      precision: c.metrics?.precision_score,
      recall: c.metrics?.recall,
      final_epsilon: c.final_epsilon,
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'fl_metrics.json'
    a.click()
  }

  const activeCompare = compare.filter(c => selectedExpIds.includes(c.experiment_id))

  return (
    <div className="page-wrapper">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="page-title">Model Comparison</h1>
          <p className="page-subtitle">Side-by-side comparison of all 5 FL algorithms</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="btn-secondary text-xs">
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
          {activeCompare.length > 0 && (
            <button onClick={exportMetrics} className="btn-secondary text-xs">
              <Download size={13} /> Export JSON
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="glass-card p-4 mb-6 flex items-center gap-3 text-red-400 border border-red-500/30">
          <AlertCircle size={15} />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {loading ? (
        <div className="text-center py-20 text-slate-500 text-sm animate-pulse">Loading comparison data…</div>
      ) : compare.length === 0 ? (
        <div className="glass-card p-16 text-center">
          <BarChart3 size={40} className="text-slate-700 mx-auto mb-4" />
          <p className="text-slate-500">No completed experiments yet</p>
          <p className="text-slate-600 text-sm mt-1">Run at least one training job to see comparisons</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Selection card */}
          <div className="glass-card p-5">
            <h2 className="text-sm font-semibold text-slate-200 mb-3">Select Experiments to Compare</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {compare.map(c => {
                const isSelected = selectedExpIds.includes(c.experiment_id)
                const dateStr = c.created_at ? new Date(c.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : ''
                return (
                  <label
                    key={c.experiment_id}
                    className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-200
                      ${isSelected ? 'bg-brand-600/10 border-brand-500/40' : 'border-white/5 hover:border-white/15 hover:bg-white/3'}`}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => {
                        setSelectedExpIds(prev =>
                          prev.includes(c.experiment_id)
                            ? prev.filter(id => id !== c.experiment_id)
                            : [...prev, c.experiment_id]
                        )
                      }}
                      className="mt-1 accent-brand-500 rounded"
                    />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`badge-${c.algorithm}`}>
                          {ALGO_META[c.algorithm]?.label || c.algorithm}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {c.experiment_id.slice(0, 5)}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 mt-1 truncate">
                        Dataset: {c.dataset_filename || 'Unknown'}
                      </p>
                      <p className="text-[10px] text-slate-500 mt-0.5">
                        {dateStr}
                      </p>
                    </div>
                  </label>
                )
              })}
            </div>
          </div>

          {activeCompare.length === 0 ? (
            <div className="glass-card p-12 text-center text-slate-500 text-sm">
              Please select at least one experiment above to display comparison details
            </div>
          ) : (
            <>
              {/* Metric bars */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {['accuracy', 'f1', 'auc', 'precision_score'].map(m => (
                  <MetricBar key={m} label={m} compare={activeCompare} />
                ))}
              </div>

              {/* Convergence + Feature Importance */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <ConvergenceChart compare={activeCompare} />
                <FeatureImportanceChart compare={activeCompare} />
              </div>

              {/* Confusion matrices */}
              <div className="glass-card p-5">
                <p className="text-sm font-semibold text-slate-200 mb-4">Confusion Matrices</p>
                <div className="flex flex-wrap gap-8">
                  {activeCompare.map(c => (
                    <ConfusionMatrix key={c.experiment_id} cm={c.metrics?.confusion_matrix} algorithm={getExpLabel(c)} />
                  ))}
                </div>
              </div>

              {/* Privacy-utility */}
              <PrivacyUtilityChart privacyCurve={privacyCurve} />

              {/* Summary table */}
              <div className="glass-card overflow-x-auto">
                <div className="px-5 py-4 border-b border-white/10">
                  <p className="text-sm font-semibold text-slate-200">Performance Summary</p>
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Experiment ID</th>
                      <th>Dataset</th>
                      <th>Accuracy</th>
                      <th>F1</th>
                      <th>AUC</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>Final ε</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeCompare.map(c => {
                      const m = c.metrics || {}
                      const meta = ALGO_META[c.algorithm] || { label: c.algorithm }
                      return (
                        <tr key={c.experiment_id}>
                          <td>
                            <span className={`badge-${c.algorithm}`}>{meta.label}</span>
                          </td>
                          <td className="font-mono text-slate-400 text-xs">
                            {c.experiment_id.slice(0, 8)}...
                          </td>
                          <td className="text-slate-300 text-xs">
                            {c.dataset_filename || '—'}
                          </td>
                          {['accuracy', 'f1', 'auc', 'precision_score', 'recall'].map(k => (
                            <td key={k} className="font-mono text-slate-200">
                              {m[k] !== undefined ? `${(m[k] * 100).toFixed(1)}%` : '—'}
                            </td>
                          ))}
                          <td className="font-mono text-rose-400">
                            {c.final_epsilon ? c.final_epsilon.toFixed(3) : '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
