import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Database, Brain, Zap, BarChart3,
  LogOut, Lock, ChevronRight, MessageSquare
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/datasets',  icon: Database,        label: 'Datasets'  },
  { to: '/training',  icon: Brain,            label: 'Training'  },
  { to: '/predict',   icon: Zap,              label: 'Predict'   },
  { to: '/agent',     icon: MessageSquare,    label: 'AI Agent'  },
  { to: '/compare',   icon: BarChart3,        label: 'Compare'   },
]

export default function Sidebar({ user, onLogout }) {

  // Display name: prefer Google name, fallback to email
  const displayName = user?.name || user?.email?.split('@')[0] || 'User'
  const avatarUrl = user?.avatar || null

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col bg-surface-800/80 backdrop-blur-md border-r border-white/10 h-screen sticky top-0">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-purple-600 flex items-center justify-center shadow-lg shadow-brand-900/50">
            <Brain size={18} className="text-white" />
          </div>
          <div>
            <p className="font-bold text-slate-100 leading-tight text-sm">FL Platform</p>
            <p className="text-xs text-slate-500">Privacy-Preserving ML</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
               transition-all duration-200 group
               ${isActive
                 ? 'bg-brand-600/20 text-brand-400 border border-brand-500/30'
                 : 'text-slate-400 hover:text-slate-100 hover:bg-white/5 border border-transparent'
               }`
            }
          >
            <Icon size={16} className="flex-shrink-0" />
            <span>{label}</span>
            <ChevronRight size={12} className="ml-auto opacity-0 group-hover:opacity-50 transition-opacity" />
          </NavLink>
        ))}
      </nav>

      {/* Encryption badge */}
      <div className="px-4 py-3 mx-3 mb-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
        <div className="flex items-center gap-2">
          <Lock size={12} className="text-emerald-400" />
          <span className="text-xs text-emerald-400 font-medium">AES-256-GCM Active</span>
          <div className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </div>
        <p className="text-xs text-emerald-600 mt-0.5">RSA-2048 key exchange</p>
      </div>

      {/* User */}
      <div className="px-4 py-4 border-t border-white/10">
        <div className="flex items-center gap-3">
          {avatarUrl ? (
            <img
              src={avatarUrl}
              alt={displayName}
              className="w-8 h-8 rounded-full flex-shrink-0 object-cover border border-white/20"
              referrerPolicy="no-referrer"
            />
          ) : (
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
              {displayName[0]?.toUpperCase() || 'U'}
            </div>
          )}
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-slate-200 truncate">{displayName}</p>
            <p className="text-xs text-slate-500 truncate">{user?.email || ''}</p>
          </div>
          <button
            onClick={onLogout}
            className="text-slate-500 hover:text-red-400 transition-colors"
            title="Logout"
          >
            <LogOut size={15} />
          </button>
        </div>
      </div>
    </aside>
  )
}
