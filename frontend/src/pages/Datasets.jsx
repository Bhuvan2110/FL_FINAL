import { useState, useCallback, useEffect } from 'react'
import { apiUpload, apiFetch } from '../utils/apiFetch'
import { Upload, Database, Eye, Trash2, FileSpreadsheet, AlertCircle, Loader2, User } from 'lucide-react'

function ColumnBadge({ col }) {
  const isNum = col.dtype === 'numeric'
  return (
    <div className="flex flex-col gap-0.5 p-3 rounded-xl bg-surface-700/50 border border-white/5 text-xs">
      <span className="font-mono font-medium text-slate-200 truncate">{col.column}</span>
      <span className={`text-xs ${isNum ? 'text-blue-400' : 'text-purple-400'}`}>{col.dtype}</span>
      <div className="flex justify-between mt-1 text-slate-500">
        <span>miss: {col.missing_pct}%</span>
        <span>uniq: {col.unique_count}</span>
      </div>
      {isNum && col.mean !== undefined && (
        <span className="text-slate-600 font-mono">μ={col.mean.toFixed(2)}</span>
      )}
    </div>
  )
}

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [preview, setPreview] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const loadDatasets = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiFetch('/dataset/list')
      setDatasets(Array.isArray(data) ? data : [])
    } catch (e) {
      // Don't block the page — just show an inline error
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadDatasets() }, [loadDatasets])

  const handleFile = async (file) => {
    if (!file) return
    if (!file.name.endsWith('.csv')) { setError('Only CSV files accepted'); return }

    // Optimistic: add a placeholder entry immediately so it feels instant
    const tempId = `temp-${Date.now()}`
    const optimisticEntry = {
      id: tempId,
      filename: file.name,
      row_count: null,
      cols: [],
      uploaded_by: null,
      created_at: new Date().toISOString(),
      _uploading: true,
    }
    setDatasets(prev => [optimisticEntry, ...prev])
    setUploading(true)
    setError(null)

    try {
      const form = new FormData()
      form.append('file', file)
      const result = await apiUpload('/dataset/upload', form)
      // Replace the optimistic entry with real data
      setDatasets(prev =>
        prev.map(d => d.id === tempId
          ? {
              id: result.id,
              filename: result.filename,
              uploaded_by: result.uploaded_by || null,
              row_count: result.row_count,
              cols: result.columns,
              created_at: new Date().toISOString(),
            }
          : d
        )
      )
    } catch (e) {
      // Remove optimistic entry on failure
      setDatasets(prev => prev.filter(d => d.id !== tempId))
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const handlePreview = async (id) => {
    if (id.startsWith('temp-')) return // Can't preview uploading file
    try {
      const data = await apiFetch(`/dataset/preview/${id}`)
      setPreview(data)
    } catch (e) {
      setError(e.message)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Delete this dataset?')) return
    // Optimistic removal
    setDatasets(prev => prev.filter(d => d.id !== id))
    if (preview) setPreview(null)
    try {
      await apiFetch(`/dataset/${id}`, { method: 'DELETE' })
    } catch (e) {
      // Re-fetch on failure to restore state
      setError(e.message)
      loadDatasets()
    }
  }

  return (
    <div className="page-wrapper">
      <h1 className="page-title">Datasets</h1>
      <p className="page-subtitle">Upload CSV datasets to Supabase Storage for FL training</p>

      {/* Drop zone */}
      <div
        className={`drop-zone mb-6 ${dragOver ? 'active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('csv-input').click()}
      >
        <input id="csv-input" type="file" accept=".csv" className="hidden" onChange={e => handleFile(e.target.files[0])} />
        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 size={32} className="text-brand-400 animate-spin" />
            <p className="text-slate-400 text-sm">Uploading and profiling…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-brand-500/10 border border-brand-500/30 flex items-center justify-center">
              <Upload size={24} className="text-brand-400" />
            </div>
            <div>
              <p className="text-slate-300 font-medium">Drop CSV here or click to browse</p>
              <p className="text-slate-500 text-sm mt-1">Binary classification target in last column · Numeric features only</p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="glass-card p-4 mb-4 flex items-center justify-between gap-3 border border-red-500/30">
          <div className="flex items-center gap-3 text-red-400 min-w-0">
            <AlertCircle size={15} className="flex-shrink-0" />
            <span className="text-sm truncate">
              {error.includes('Cannot reach') || error.includes('waking')
                ? '⏳ Backend is waking up from sleep — retrying automatically…'
                : error}
            </span>
          </div>
          <button onClick={loadDatasets} className="btn-secondary text-xs shrink-0 ml-2">Retry</button>
        </div>
      )}

      {/* Dataset list */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {loading ? (
          <div className="col-span-2 text-center text-slate-500 py-8 text-sm animate-pulse">Loading datasets…</div>
        ) : datasets.length === 0 ? (
          <div className="col-span-2 glass-card p-10 text-center">
            <Database size={32} className="text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No datasets yet — upload your first CSV above</p>
          </div>
        ) : datasets.map(ds => (
          <div key={ds.id} className={`glass-card p-4 flex items-start justify-between gap-3 ${ds._uploading ? 'opacity-60' : ''}`}>
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center flex-shrink-0">
                {ds._uploading
                  ? <Loader2 size={16} className="text-brand-400 animate-spin" />
                  : <FileSpreadsheet size={16} className="text-brand-400" />
                }
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200 font-mono">{ds.filename}</p>
                <p className="text-xs text-slate-500">
                  {ds._uploading ? 'Uploading…' : `${ds.row_count?.toLocaleString()} rows · ${ds.cols?.length} cols`}
                </p>
                {/* Show uploader's name (Google account name) */}
                {ds.uploaded_by && (
                  <p className="text-xs text-brand-400/80 mt-0.5 flex items-center gap-1">
                    <User size={10} className="flex-shrink-0" />
                    <span className="truncate">{ds.uploaded_by}</span>
                  </p>
                )}
                <p className="text-xs text-slate-600 mt-0.5">{new Date(ds.created_at).toLocaleDateString()}</p>
              </div>
            </div>
            {!ds._uploading && (
              <div className="flex items-center gap-2">
                <button onClick={() => handlePreview(ds.id)} className="btn-secondary text-xs py-1.5 px-3">
                  <Eye size={12} /> Preview
                </button>
                <button onClick={() => handleDelete(ds.id)} className="btn-danger text-xs py-1.5 px-3">
                  <Trash2 size={12} />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Preview panel */}
      {preview && (
        <div className="glass-card p-5 animate-slide-up">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Eye size={14} className="text-brand-400" /> Column Profile
            </h2>
            <button onClick={() => setPreview(null)} className="text-xs text-slate-500 hover:text-slate-300">✕ Close</button>
          </div>

          {/* Column cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 mb-5">
            {(preview.col_profiles || []).map(col => <ColumnBadge key={col.column} col={col} />)}
          </div>

          {/* Data preview table */}
          <div className="overflow-x-auto">
            <table className="data-table text-xs">
              <thead>
                <tr>
                  {(preview.headers || []).map(h => <th key={h}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {(preview.rows || []).slice(0, 8).map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => <td key={j} className="font-mono">{cell}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-600 mt-2">Showing first 8 of {preview.total_rows} rows</p>
        </div>
      )}
    </div>
  )
}
