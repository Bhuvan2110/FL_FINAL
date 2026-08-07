import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { supabase } from './lib/supabaseClient'
import { setAuthToken, wakeBackend, startKeepAlive, stopKeepAlive } from './utils/apiFetch'
import Sidebar from './components/Sidebar'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Training from './pages/Training'
import Predict from './pages/Predict'
import Comparison from './pages/Comparison'
import AgentChat from './pages/AgentChat'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Wake the backend from Render free-tier sleep (runs in parallel with auth check)
    wakeBackend()

    // Default fallback user for instant accessibility
    const fallbackUser = {
      id: 'mock-google-id',
      email: 'google-user@example.com',
      name: 'Google User',
      avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop&crop=faces',
      role: 'user',
      token: 'mock-google-token',
    }

    // Check existing session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        setAuthToken(session.access_token)
        const u = session.user
        const userData = {
          id: u.id,
          email: u.email,
          name: u.user_metadata?.full_name || u.user_metadata?.name || u.email?.split('@')[0] || 'User',
          avatar: u.user_metadata?.avatar_url || null,
          role: u.app_metadata?.role || 'user',
          token: session.access_token,
        }
        setUser(userData)
        localStorage.setItem('fl_user', JSON.stringify(userData))
      } else {
        const savedUserStr = localStorage.getItem('fl_user')
        if (savedUserStr) {
          try {
            const savedUser = JSON.parse(savedUserStr)
            setAuthToken(savedUser.token || 'mock-google-token')
            setUser(savedUser)
          } catch (_) {
            setAuthToken(fallbackUser.token)
            setUser(fallbackUser)
          }
        } else {
          setAuthToken(fallbackUser.token)
          setUser(fallbackUser)
        }
      }
      setLoading(false)
    }).catch(() => {
      const savedUserStr = localStorage.getItem('fl_user')
      if (savedUserStr) {
        try {
          const savedUser = JSON.parse(savedUserStr)
          setAuthToken(savedUser.token || 'mock-google-token')
          setUser(savedUser)
        } catch (_) {
          setAuthToken(fallbackUser.token)
          setUser(fallbackUser)
        }
      } else {
        setAuthToken(fallbackUser.token)
        setUser(fallbackUser)
      }
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_OUT') {
        localStorage.removeItem('fl_user')
        setUser(null)
        setAuthToken(null)
      } else if (session) {
        setAuthToken(session.access_token)
        const u = session.user
        const userData = {
          id: u.id,
          email: u.email,
          name: u.user_metadata?.full_name || u.user_metadata?.name || u.email?.split('@')[0] || 'User',
          avatar: u.user_metadata?.avatar_url || null,
          role: u.app_metadata?.role || 'user',
          token: session.access_token,
        }
        setUser(userData)
        localStorage.setItem('fl_user', JSON.stringify(userData))
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  // Start/stop keep-alive based on auth state
  useEffect(() => {
    if (user) {
      startKeepAlive()
    } else {
      stopKeepAlive()
    }
    return () => stopKeepAlive()
  }, [user])

  const handleLogin = (userData) => {
    setAuthToken(userData.token || 'mock-google-token')
    setUser(userData)
    localStorage.setItem('fl_user', JSON.stringify(userData))
  }

  const handleSkipLogin = () => {
    const guestUser = {
      id: 'guest',
      email: 'guest@demo.local',
      name: 'Guest User',
      role: 'guest',
      token: 'guest-token',
    }
    setAuthToken('guest-token')
    setUser(guestUser)
    localStorage.setItem('fl_user', JSON.stringify(guestUser))
  }

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut()
    } catch (_) {}
    localStorage.removeItem('fl_user')
    setUser(null)
    setAuthToken(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 rounded-full border-2 border-brand-500/30 border-t-brand-500 animate-spin" />
      </div>
    )
  }

  if (!user) {
    return <Login onLogin={handleLogin} onSkip={handleSkipLogin} />
  }

  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden">
        <Sidebar user={user} onLogout={handleLogout} />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/datasets"  element={<Datasets />} />
            <Route path="/training"  element={<Training />} />
            <Route path="/predict"   element={<Predict />} />
            <Route path="/agent"     element={<AgentChat />} />
            <Route path="/compare"   element={<Comparison />} />
            <Route path="*"          element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
