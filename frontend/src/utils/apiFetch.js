const API_BASE = import.meta.env.VITE_API_URL || 'https://fl-platform-api-kg2m.onrender.com'

let _token = null

export function setAuthToken(token) {
  _token = token
}

export function getAuthToken() {
  return _token
}

// Ping /health to wake Render free-tier from sleep
export async function wakeBackend() {
  try {
    await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(65000) })
  } catch (_) {
    // silently ignore — just a warm-up call
  }
}

// Keep backend alive while the app tab is open (ping every 10 minutes)
let _keepAliveTimer = null
export function startKeepAlive() {
  if (_keepAliveTimer) return
  _keepAliveTimer = setInterval(() => {
    fetch(`${API_BASE}/health`).catch(() => {})
  }, 10 * 60 * 1000) // 10 minutes
}
export function stopKeepAlive() {
  if (_keepAliveTimer) {
    clearInterval(_keepAliveTimer)
    _keepAliveTimer = null
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 60000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('timeout')
    }
    throw new Error('network')
  } finally {
    clearTimeout(timer)
  }
}

// Retry up to `maxRetries` times, waiting `delayMs` between attempts
export async function apiFetch(path, options = {}, maxRetries = 1, delayMs = 1500) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }
  if (_token) headers['Authorization'] = `Bearer ${_token}`

  let lastErr
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeout(`${API_BASE}${path}`, { ...options, headers }, 60000)
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(body.detail || `API error ${res.status}`)
      }
      return await res.json()
    } catch (err) {
      lastErr = err
      // Only retry on network/timeout errors, not API errors (4xx/5xx)
      const isNetworkErr = err.message === 'timeout' || err.message === 'network'
      if (!isNetworkErr || attempt === maxRetries) break
      // Wait before retrying
      await new Promise(r => setTimeout(r, delayMs))
    }
  }

  // Map internal codes to user-friendly messages
  if (lastErr.message === 'timeout' || lastErr.message === 'network') {
    throw new Error('Cannot reach server — the backend may be starting up. Please retry in a moment.')
  }
  throw lastErr
}

export async function apiUpload(path, formData, maxRetries = 1, delayMs = 2000) {
  const headers = {}
  if (_token) headers['Authorization'] = `Bearer ${_token}`

  let lastErr
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeout(
        `${API_BASE}${path}`,
        { method: 'POST', headers, body: formData },
        60000 // uploads can take longer
      )
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(body.detail || `Upload error ${res.status}`)
      }
      return await res.json()
    } catch (err) {
      lastErr = err
      const isNetworkErr = err.message === 'timeout' || err.message === 'network'
      if (!isNetworkErr || attempt === maxRetries) break
      await new Promise(r => setTimeout(r, delayMs))
    }
  }

  if (lastErr.message === 'timeout' || lastErr.message === 'network') {
    throw new Error('Cannot reach server — the backend may be starting up. Please retry in a moment.')
  }
  throw lastErr
}
