import { useState, useEffect } from 'react'
import { apiFetch } from '../utils/apiFetch'
import {
  Brain, BarChart3, CheckCircle2,
  TrendingUp, Activity, RefreshCw, AlertCircle, Cpu
} from 'lucide-react'

const ALGO_COLORS = {
  fedavg:   { text: 'text-blue-400',    bg: 'bg-blue-500/15',    border: 'border-blue-500/30'   },
  fedprox:  { text: 'text-purple-400',  bg: 'bg-purple-500/15',  border: 'border-purple-500/30' },
  scaffold: { text: 'text-emerald-400', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30'},
  dpsgd:    { text: 'text-rose-400',    bg: 'bg-rose-500/15',    border: 'border-rose-500/30'   },
  central:  { text: 'text-amber-400',   bg: 'bg-amber-500/15',   border: 'border-amber-500/30'  },
}

function StatCard({ icon: Icon, label, value, sub, color = 'brand' }) {
  return (
    <div className="stat-card">
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center mb-2
        ${color === 'brand'   ? 'bg-brand-500/15'   : ''}
        ${color === 'green'   ? 'bg-emerald-500/15' : ''}
        ${color === 'purple'  ? 'bg-purple-500/15'  : ''}
        ${color === 'amber'   ? 'bg-amber-500/15'   : ''}
      `}>
        <Icon size={16} className={
          color === 'brand'  ? 'text-brand-400'   :
          color === 'green'  ? 'text-emerald-400' :
          color === 'purple' ? 'text-purple-400'  : 'text-amber-400'
        } />
      </div>
      <p className="text-2xl font-bold text-slate-100">{value}</p>
      <p className="text-xs font-medium text-slate-400">{label}</p>
      {sub && <p className="text-xs text-slate-600 mt-0.5">{sub}</p>}
    </div>
  )
}

function StatusBadge({ status }) {
  const cls = {
    pending:   'badge-pending',
    running:   'badge-running',
    completed: 'badge-completed',
    failed:    'badge-failed',
  }
  return <span className={cls[status] || 'badge-pending'}>{status}</span>
}

function AlgoBadge({ algo }) {
  const c = ALGO_COLORS[algo] || ALGO_COLORS.fedavg
  return (
    <span className={`px-2 py-0.5 rounded-md text-xs font-mono ${c.bg} ${c.text} border ${c.border}`}>
      {algo}
    </span>
  )
}

export default function Dashboard() {
  const [experiments, setExperiments] = useState([])
  const [compare, setCompare] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = async () => {
    try {
      setLoading(true)
      setError(null)
      const [exps, cmp] = await Promise.allSettled([
        apiFetch('/training/list'),
        apiFetch('/training/compare'),
      ])
      setExperiments(exps.status === 'fulfilled' && Array.isArray(exps.value) ? exps.value : [])
      setCompare(cmp.status === 'fulfilled' && Array.isArray(cmp.value) ? cmp.value : [])
      // Show error only if both failed
      if (exps.status === 'rejected' && cmp.status === 'rejected') {
        setError(exps.reason?.message || 'Failed to load data')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const completed = experiments.filter(e => e.status === 'completed').length
  const running   = experiments.filter(e => e.status === 'running').length
  const bestAcc   = compare.reduce((best, c) => Math.max(best, c.metrics?.accuracy || 0), 0)
  const bestAlgo  = compare.find(c => c.metrics?.accuracy === bestAcc)?.algorithm

  return (
    <div className="page-wrapper">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">Overview of your federated learning experiments</p>
        </div>
        <button onClick={load} className="btn-secondary text-xs gap-1.5">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard icon={Brain}       label="Experiments"  value={experiments.length} color="brand"  />
        <StatCard icon={CheckCircle2} label="Completed"   value={completed}          color="green"  />
        <StatCard icon={Activity}    label="Running"      value={running}            color="purple" sub={running > 0 ? 'Training active' : 'Idle'} />
        <StatCard icon={TrendingUp}  label="Best Accuracy" value={bestAcc ? `${(bestAcc*100).toFixed(1)}%` : '—'} color="amber" sub={bestAlgo} />
      </div>

      {error && (
        <div className="glass-card p-4 mb-6 flex items-center justify-between gap-3 border border-red-500/30">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle size={16} className="flex-shrink-0" />
            <span className="text-sm">
              {error.includes('waking') || error.includes('Cannot reach') || error.includes('timed out')
                ? '⏳ Backend is waking up from sleep — please retry in a moment.'
                : error}
            </span>
          </div>
          <button onClick={load} className="btn-secondary text-xs shrink-0">Retry</button>
        </div>
      )}

      {/* Model Summary */}
      {compare.length > 0 && (
        <div className="glass-card p-5 mb-6">
          <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <BarChart3 size={15} className="text-brand-400" />
            Model Performance Summary
          </h2>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
            {compare.map(c => {
              const col = ALGO_COLORS[c.algorithm] || ALGO_COLORS.fedavg
              return (
                <div key={c.experiment_id} className={`rounded-xl p-3 border ${col.bg} ${col.border}`}>
                  <p className={`text-xs font-mono font-bold uppercase ${col.text} mb-2`}>{c.algorithm}</p>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Acc</span>
                      <span className="text-slate-200 font-mono">{((c.metrics?.accuracy || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">F1</span>
                      <span className="text-slate-200 font-mono">{((c.metrics?.f1 || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">AUC</span>
                      <span className="text-slate-200 font-mono">{(c.metrics?.auc || 0).toFixed(3)}</span>
                    </div>
                    {c.final_epsilon && (
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-500">ε</span>
                        <span className="text-rose-400 font-mono">{c.final_epsilon.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Recent experiments */}
      <div className="glass-card">
        <div className="px-5 py-4 border-b border-white/10 flex items-center gap-2">
          <Cpu size={15} className="text-brand-400" />
          <span className="text-sm font-semibold text-slate-200">Recent Experiments</span>
          <span className="ml-auto text-xs text-slate-500">{experiments.length} total</span>
        </div>
        {loading ? (
          <div className="p-12 text-center text-slate-500 text-sm animate-pulse">Loading…</div>
        ) : experiments.length === 0 ? (
          <div className="p-12 text-center">
            <Brain size={32} className="text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No experiments yet</p>
            <p className="text-slate-600 text-xs mt-1">Upload a dataset and start training</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Dataset</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {experiments.slice(0, 10).map(exp => (
                  <tr key={exp.id}>
                    <td><AlgoBadge algo={exp.algorithm} /></td>
                    <td className="text-slate-400 text-xs font-mono">{exp.datasets?.filename || '—'}</td>
                    <td><StatusBadge status={exp.status} /></td>
                    <td className="text-slate-500 text-xs">{new Date(exp.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
