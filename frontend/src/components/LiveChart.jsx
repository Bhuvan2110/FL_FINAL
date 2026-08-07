import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { supabase } from '../lib/supabaseClient'
import { TrendingUp, Activity } from 'lucide-react'

const ALGO_COLORS = {
  fedavg:   '#3b82f6',
  fedprox:  '#8b5cf6',
  scaffold: '#10b981',
  dpsgd:    '#f43f5e',
  central:  '#f59e0b',
}

export default function LiveChart({ experimentId, algorithm }) {
  const [rounds, setRounds] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!experimentId) return

    // Initial load
    supabase
      .from('rounds')
      .select('round_num, loss, accuracy, val_accuracy')
      .eq('experiment_id', experimentId)
      .order('round_num')
      .then(({ data }) => {
        if (data) setRounds(data.map(r => ({ ...r, round: r.round_num })))
        setLoading(false)
      })

    // Realtime subscription
    const channel = supabase
      .channel(`rounds-${experimentId}`)
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'rounds',
        filter: `experiment_id=eq.${experimentId}`,
      }, (payload) => {
        setRounds(prev => [...prev, { ...payload.new, round: payload.new.round_num }])
      })
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [experimentId])

  const color = ALGO_COLORS[algorithm] || '#3b82f6'
  const latestAcc = rounds.length > 0 ? (rounds[rounds.length - 1].val_accuracy ?? rounds[rounds.length - 1].accuracy) : null

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-brand-400" />
          <span className="text-sm font-semibold text-slate-200">Live Training Progress</span>
          {rounds.length > 0 && (
            <span className="text-xs text-slate-500">Round {rounds.length}</span>
          )}
        </div>
        {latestAcc !== null && (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-brand-500/15 border border-brand-500/30">
            <TrendingUp size={12} className="text-brand-400" />
            <span className="text-xs font-mono text-brand-400">{(latestAcc * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>

      {loading ? (
        <div className="h-48 flex items-center justify-center text-slate-500 text-sm">
          <div className="animate-pulse">Loading chart…</div>
        </div>
      ) : rounds.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-slate-500 text-sm">
          Waiting for training to start…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rounds} margin={{ top: 0, right: 10, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="round" tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#64748b' }} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, color: '#64748b' }} />
            <Line type="monotone" dataKey="accuracy" stroke={color} strokeWidth={2} dot={false} name="Train Acc" />
            <Line type="monotone" dataKey="val_accuracy" stroke={color} strokeWidth={2} strokeDasharray="4 4" dot={false} name="Val Acc" />
            <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={1.5} dot={false} name="Loss" opacity={0.6} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
