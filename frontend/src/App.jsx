import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const IPFS_GATEWAY_BASE = 'https://gateway.pinata.cloud/ipfs/'
const SEPOLIA_TX_BASE = 'https://sepolia.etherscan.io/tx/'
const AUTH_TOKEN_KEY = 'auth_token'

const formatPercent = (value) => `${Math.round(value * 100)}%`
const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

const LABEL_HELP = {
  NV: 'Benign mole, common skin mark',
  MEL: 'Malignant melanoma, needs evaluation',
  BKL: 'Benign keratosis-like skin growth',
  BCC: 'Basal cell carcinoma, slow-growing',
  AKIEC: 'Precancerous sun-damaged lesion',
  VASC: 'Blood vessel-related skin lesion',
  DF: 'Firm benign fibrous skin nodule'
}

const DISPLAY_LABELS = {
  NV: 'Melanocytic nevus (benign mole)',
  MEL: 'Melanoma',
  BKL: 'Benign keratosis-like lesion',
  BCC: 'Basal cell carcinoma',
  AKIEC: 'Actinic keratosis / intraepithelial carcinoma',
  VASC: 'Vascular lesion',
  DF: 'Dermatofibroma'
}

const DISEASE_DETAILS = {
  NV: 'Melanocytic nevus is usually a benign mole. Watch for sudden ABCDE changes and consult a clinician if it evolves.',
  MEL: 'Melanoma is a potentially dangerous skin cancer. Early clinical evaluation is important for suspicious lesions.',
  BKL: 'Benign keratosis-like lesions are often non-cancerous but may look similar to other conditions.',
  BCC: 'Basal cell carcinoma is a common slow-growing skin cancer and should be confirmed by a dermatologist.',
  AKIEC: 'Actinic keratosis / intraepithelial carcinoma can be precancerous and needs medical review.',
  VASC: 'Vascular lesions are related to blood vessels. Most are benign, but persistent changes should be checked.',
  DF: 'Dermatofibroma is usually a benign fibrous skin nodule and is often stable over time.',
  acne: 'Acne is a follicle and oil-gland condition that may present with comedones, papules, or pustules.',
  eczema: 'Eczema often causes dry, itchy, inflamed skin and can flare with irritants or allergens.',
  psoriasis: 'Psoriasis is a chronic inflammatory skin condition with red, scaly plaques.',
  rosacea: 'Rosacea often causes persistent facial redness, sensitivity, and acne-like bumps.'
}

const normalizeKey = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
const formatLabel = (label) => DISPLAY_LABELS[label] || label.replace(/_/g, ' ')

const getDiseaseInfo = (label) => {
  const normalized = normalizeKey(label)
  const known = DISEASE_DETAILS[label] || DISEASE_DETAILS[normalized] || LABEL_HELP[label]
  if (known) return known
  return `${formatLabel(label)} is a model-predicted skin condition label. This result is educational only, not a diagnosis. If symptoms are persistent, painful, bleeding, or worsening, seek in-person evaluation from a dermatologist.`
}

const normalizePredictResponse = (data) => {
  const safeTop5 = Array.isArray(data?.top5) ? data.top5 : []
  const safeTop3 = Array.isArray(data?.top3) && data.top3.length > 0 ? data.top3 : safeTop5.slice(0, 3)

  const topPrediction =
    typeof data?.prediction === 'object' && data?.prediction !== null
      ? data.prediction
      : safeTop5[0] || {
          index: -1,
          label: typeof data?.prediction === 'string' ? data.prediction : 'Unknown',
          probability: 0
        }

  const top5Combined =
    typeof data?.top5_combined === 'number'
      ? data.top5_combined
      : safeTop5.slice(0, 5).reduce((sum, item) => sum + (Number(item?.probability) || 0), 0)

  const modelMeta =
    data?.model && typeof data.model === 'object'
      ? data.model
      : {
          name: data?.used_model || 'unknown',
          version: data?.model_version || 'v1.0'
        }

  return {
    ...data,
    prediction: topPrediction,
    top3: safeTop3,
    top5: safeTop5,
    top5_combined: top5Combined,
    model: modelMeta
  }
}

const readReportsFromStorage = () => {
  try {
    const raw = localStorage.getItem('reports')
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

const isReportReady = (report) =>
  !!(report?.json_cid && report?.pdf_cid && report?.hash && report?.tx_hash)



export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [modalityOverride, setModalityOverride] = useState('auto')
  const [theme, setTheme] = useState('light')
  const [quality, setQuality] = useState({
    face: 'unknown',
    lighting: 'unknown',
    blur: 'unknown'
  })
  const [history, setHistory] = useState([])
  const [skinCloseupWarning, setSkinCloseupWarning] = useState(false)
  const [diseaseInfoOpen, setDiseaseInfoOpen] = useState(null)
  const [showFullHistory, setShowFullHistory] = useState(false)
  const [openHistoryRows, setOpenHistoryRows] = useState({})
  const [authMode, setAuthMode] = useState('login')
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(AUTH_TOKEN_KEY) || '')
  const [authUser, setAuthUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(false)
  const [authError, setAuthError] = useState('')
  const [verificationByReportId, setVerificationByReportId] = useState({})
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const previousUserIdRef = useRef(null)
  const [reports, setReports] = useState(() => readReportsFromStorage())

  const resetAnalysisView = () => {
    if (preview) {
      URL.revokeObjectURL(preview)
    }
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
    setHistory([])
    setSkinCloseupWarning(false)
    setDiseaseInfoOpen(null)
    setOpenHistoryRows({})
  }

  const fetchMyReports = async (token) => {
    if (!token) return
    const response = await fetch(`${API_URL}/api/reports`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || 'Unable to load reports')
    }
    const rows = Array.isArray(data?.reports) ? data.reports : []
    setReports(rows)
    localStorage.setItem('reports', JSON.stringify(rows))
  }

  const loadAuthProfile = async (token) => {
    const response = await fetch(`${API_URL}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || 'Session expired')
    }
    setAuthUser(data.user || null)
    await fetchMyReports(token)
  }

  const handleAuthSubmit = async (event) => {
    event.preventDefault()
    setAuthLoading(true)
    setAuthError('')
    try {
      const endpoint = authMode === 'register' ? '/api/auth/register' : '/api/auth/login'
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: authEmail, password: authPassword })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed')
      }
      const token = data?.token || ''
      setAuthToken(token)
      localStorage.setItem(AUTH_TOKEN_KEY, token)
      setAuthUser(data?.user || null)
      resetAnalysisView()
      await fetchMyReports(token)
      setAuthPassword('')
    } catch (err) {
      setAuthError(err.message || 'Authentication failed')
    } finally {
      setAuthLoading(false)
    }
  }

  const handleLogout = () => {
    resetAnalysisView()
    setAuthToken('')
    setAuthUser(null)
    setAuthPassword('')
    setAuthError('')
    localStorage.removeItem(AUTH_TOKEN_KEY)
    const localReports = readReportsFromStorage()
    setReports(localReports)
  }

  const saveReportMetadata = (normalized) => {
  if (authToken) return

  const entry = {
    reportId: normalized?.reportId,
    json_cid: normalized?.json_cid,
    pdf_cid: normalized?.pdf_cid,
    hash: normalized?.hash,
    tx_hash: normalized?.tx_hash,
    prediction: normalized?.prediction?.label || '',
    model: normalized?.model || null,
    timestamp: normalized?.timestamp || new Date().toISOString()
  }

  const required =
    entry.reportId &&
    entry.json_cid &&
    entry.pdf_cid &&
    entry.hash &&
    entry.tx_hash &&
    entry.prediction &&
    entry.model &&
    entry.timestamp

  if (!required) return

  setReports((prev) => {
    const next = [entry, ...prev.filter((item) => item.reportId !== entry.reportId)]
    localStorage.setItem('reports', JSON.stringify(next))
    return next
  })
}

  const toggleHistoryRow = (reportId) => {
    setOpenHistoryRows((prev) => ({
      ...prev,
      [reportId]: !prev[reportId]
    }))
  }

  const verifyHistoryReport = async (entry) => {
    const reportId = entry?.reportId
    const jsonCid = entry?.json_cid
    if (!reportId) return

    if (!jsonCid) {
      setVerificationByReportId((prev) => ({
        ...prev,
        [reportId]: {
          status: 'error',
          message: 'Cannot verify this report because JSON CID is missing.'
        }
      }))
      return
    }

    setVerificationByReportId((prev) => ({
      ...prev,
      [reportId]: {
        status: 'loading',
        message: 'Verifying report integrity...'
      }
    }))

    try {
      const response = await fetch(`${API_URL}/verify/${encodeURIComponent(jsonCid)}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Verification failed.')
      }

      const status = String(data?.status || '').toLowerCase()
      if (status === 'verified') {
        setVerificationByReportId((prev) => ({
          ...prev,
          [reportId]: {
            status: 'verified',
            message: 'Verified'
          }
        }))
      } else {
        setVerificationByReportId((prev) => ({
          ...prev,
          [reportId]: {
            status: 'tampered',
            message: 'Tampered: the recomputed JSON hash does not match the on-chain record.'
          }
        }))
      }
    } catch (err) {
      setVerificationByReportId((prev) => ({
        ...prev,
        [reportId]: {
          status: 'error',
          message: err.message || 'Verification failed.'
        }
      }))
    }
  }

  useEffect(() => {
    if (!authToken) return
    loadAuthProfile(authToken).catch(() => {
      handleLogout()
    })
  }, [authToken])

  useEffect(() => {
    const currentUserId = authUser?.id || null
    if (!currentUserId) {
      previousUserIdRef.current = null
      return
    }

    if (previousUserIdRef.current && previousUserIdRef.current !== currentUserId) {
      // Prevent previous account's diagnosis/report state from leaking into another account session.
      resetAnalysisView()
    }

    previousUserIdRef.current = currentUserId
  }, [authUser?.id])

  useEffect(() => {
    if (!result?.reportId) return
    if (result.persistence_status === 'complete' && isReportReady(result)) {
      saveReportMetadata(result)
      return
    }
    if (result.persistence_status === 'error') return

    let cancelled = false

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/report-status/${result.reportId}`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : undefined
        })
        if (!response.ok) return
        const status = await response.json()
        if (cancelled) return

        setResult((current) => ({
          ...current,
          ...status,
          persistence_status: status.persistence_status || current.persistence_status,
          persistence_message: status.persistence_message || current.persistence_message
        }))

        if (status.persistence_status === 'complete' && isReportReady(status)) {
          saveReportMetadata({ ...result, ...status })
          if (authToken) {
            fetchMyReports(authToken).catch(() => {})
          }
        }
      } catch {
        // Ignore transient polling failures; the next tick may succeed.
      }
    }

    pollStatus()
    const intervalId = window.setInterval(pollStatus, 3000)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [result?.reportId, result?.persistence_status, authToken])

  const statusText = useMemo(() => {
    if (loading) return 'Analyzing image...'
    if (result) return 'Result ready'
    return 'Upload a clear close-up of the affected area'
  }, [loading, result])

  const onFileChange = (event) => {
    const selected = event.target.files?.[0]
    if (!selected) return
    setFile(selected)
    setResult(null)
    setError('')
    setPreview(URL.createObjectURL(selected))
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      setCameraOn(true)
    } catch (err) {
      setError('Unable to access camera. Please allow camera permission.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraOn(false)
  }

  useEffect(() => {
    if (!cameraOn || !videoRef.current || !streamRef.current) return
    const video = videoRef.current
    video.srcObject = streamRef.current
    video.muted = true
    video.onloadedmetadata = () => {
      video
        .play()
        .catch(() => setError('Camera started but could not autoplay. Click the video to play.'))
    }
  }, [cameraOn])

  const capturePhoto = () => {
    if (!videoRef.current) return
    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (!blob) return
      const capturedFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
      setFile(capturedFile)
      setPreview(URL.createObjectURL(blob))
      setResult(null)
      setError('')
    }, 'image/jpeg', 0.92)
  }

  const assessQuality = async (blobUrl) => {
    try {
      const img = new Image()
      img.src = blobUrl
      await img.decode()
      const canvas = document.createElement('canvas')
      const size = 256
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, size, size)
      const imageData = ctx.getImageData(0, 0, size, size)
      const data = imageData.data
      let sum = 0
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]
        sum += 0.2126 * r + 0.7152 * g + 0.0722 * b
      }
      const mean = sum / (size * size)
      const lighting = mean < 60 ? 'low' : mean > 200 ? 'high' : 'good'

      // Simple blur estimation: variance of Laplacian
      let variance = 0
      let lapSum = 0
      const gray = new Float32Array(size * size)
      for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
        gray[j] = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]
      }
      for (let y = 1; y < size - 1; y += 1) {
        for (let x = 1; x < size - 1; x += 1) {
          const idx = y * size + x
          const lap =
            gray[idx - size] +
            gray[idx - 1] +
            gray[idx + 1] +
            gray[idx + size] -
            4 * gray[idx]
          lapSum += lap
          variance += lap * lap
        }
      }
      const count = (size - 2) * (size - 2)
      const lapMean = lapSum / count
      const lapVar = variance / count - lapMean * lapMean
      const blur = lapVar < 20 ? 'blur' : 'sharp'

      let face = 'unknown'
      if ('FaceDetector' in window) {
        const detector = new window.FaceDetector({ fastMode: true })
        const faces = await detector.detect(img)
        face = faces.length > 0 ? 'yes' : 'no'
        setSkinCloseupWarning(faces.length > 0)
      }

      setQuality({
        face,
        lighting,
        blur
      })
    } catch {
      setQuality({ face: 'unknown', lighting: 'unknown', blur: 'unknown' })
      setSkinCloseupWarning(false)
    }
  }

  useEffect(() => {
    if (!preview) return
    assessQuality(preview)
  }, [preview])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const onSubmit = async (event) => {
    event.preventDefault()
    if (!file) {
      setError('Please select an image.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('image', file)
    if (modalityOverride !== 'auto') {
      formData.append('modality', modalityOverride)
    }

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : undefined,
        body: formData
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }
      if (preview) {
        setHistory((prev) => [preview, ...prev].slice(0, 2))
      }
      const normalized = normalizePredictResponse(data)
      setResult(normalized)
    } catch (err) {
      setError(err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={`app ${!authUser ? 'app-auth-locked' : ''}`}>
      <header className="hero">
        <div>
          <p className="pill">AI Skin Check</p>
          <h1>Dermalyze</h1>
          <p className="subtitle">
            Upload a skin image for an educational, AI-powered assessment. This tool
            is not a medical device and cannot diagnose disease.
          </p>
          <div className="toggle-row">
            <button
              type="button"
              className="secondary-action"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            >
              {theme === 'light' ? 'Dark mode' : 'Light mode'}
            </button>
            <button
              type="button"
              className="secondary-action"
              onClick={() => {
                if (result?.pdf_cid) {
                  window.open(`${IPFS_GATEWAY_BASE}${result.pdf_cid}`, '_blank', 'noopener,noreferrer')
                  return
                }
                window.print()
              }}
            >
              {result?.pdf_cid ? 'Download report PDF' : 'Save result as PDF'}
            </button>
          </div>
        </div>
        {authUser && (
          <div className="hero-card account-card">
            <h3>Account</h3>
            <p className="hint">Logged in as {authUser.email}</p>
            <button type="button" className="secondary-action" onClick={handleLogout}>
              Logout
            </button>
          </div>
        )}
      </header>

      {!authUser && (
        <div className="auth-modal-backdrop" role="dialog" aria-modal="true" aria-label="Authentication required">
          <div className="auth-modal-card">
            <form className="auth-form auth-form-modal" onSubmit={handleAuthSubmit}>
              <h4>{authMode === 'register' ? 'Create account' : 'Login'}</h4>
              <p className="hint">Sign in to save and view your reports.</p>
              <input
                type="email"
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                placeholder="Email"
                required
              />
              <input
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="Password"
                required
              />
              <button type="submit" className="primary-action" disabled={authLoading}>
                {authLoading ? 'Please wait...' : authMode === 'register' ? 'Register' : 'Login'}
              </button>
              <button
                type="button"
                className="history-toggle"
                onClick={() => setAuthMode((prev) => (prev === 'register' ? 'login' : 'register'))}
              >
                {authMode === 'register' ? 'Already have an account? Login' : 'Need an account? Register'}
              </button>
              {authError && <p className="error">{authError}</p>}
            </form>
          </div>
        </div>
      )}

      <main className="layout">
        <section className="panel upload">
          <div className="panel-header">
            <h2>Upload image</h2>
            <span>{statusText}</span>
          </div>

          <form onSubmit={onSubmit} className="upload-form">
            <label className="upload-box">
              <input type="file" accept="image/*" onChange={onFileChange} />
              {preview ? (
                <img src={preview} alt="preview" />
              ) : (
                <div>
                  <strong>Drop an image here</strong>
                  <p className="hint">Drag & drop or click to upload. PNG, JPG, or WEBP. 6MB max.</p>
                </div>
              )}
            </label>

            <div className="modality-toggle">
              <span>Image type</span>
              <select value={modalityOverride} onChange={(e) => setModalityOverride(e.target.value)}>
                <option value="auto">Auto-detect</option>
                <option value="clinical">Clinical</option>
                <option value="dermoscopy">Dermoscopy</option>
              </select>
            </div>

            {skinCloseupWarning && (
              <div className="warning-card">
                <strong>Close-up needed</strong>
                <p>Please retake a closer image of the affected skin area. Avoid full-face selfies.</p>
              </div>
            )}

            <div className="camera-controls">
              {!cameraOn ? (
                <button type="button" className="secondary-action" onClick={startCamera}>
                  Use camera
                </button>
              ) : (
                <>
                  <button type="button" className="secondary-action" onClick={capturePhoto}>
                    Capture photo
                  </button>
                  <button type="button" className="secondary-action" onClick={stopCamera}>
                    Stop camera
                  </button>
                </>
              )}
            </div>

            {cameraOn && (
              <div className="camera-preview">
                <video ref={videoRef} autoPlay playsInline />
              </div>
            )}

            <button type="submit" className="primary-action" disabled={loading}>
              {loading ? 'Analyzing…' : 'Analyze image'}
            </button>

            <div className="support-stack">
              <div className="support-card">
                <h4>Upload tips</h4>
                <ul>
                  <li>Use even daylight (no flash).</li>
                  <li>Keep the camera 15–20 cm away.</li>
                  <li>Capture 2–3 angles for consistency.</li>
                </ul>
              </div>
              <div className="support-card">
                <h4>Session snapshot</h4>
                <div className="snapshot-row">
                  <div>
                    <strong>{reports.length}</strong>
                    <span>saved reports</span>
                  </div>
                  <div>
                    <strong>{result ? '1' : '0'}</strong>
                    <span>latest result</span>
                  </div>
                </div>
                {history.length > 0 && (
                  <div className="thumb-row">
                    {history.slice(0, 3).map((src) => (
                      <img key={src} src={src} alt="Recent upload" />
                    ))}
                  </div>
                )}
                <div className="snapshot-footer">
                  <button
                    type="button"
                    className="history-toggle"
                    onClick={() => setShowFullHistory((prev) => !prev)}
                  >
                    {showFullHistory ? 'Hide full history' : 'Show full history'}
                  </button>
                </div>

                {showFullHistory && (
                  <div className="history-list" aria-label="Saved session history">
                    {!reports.length && <p className="hint">No saved reports yet.</p>}

                    {reports.map((entry) => {
                      const key = `${entry.reportId}-${entry.timestamp}`
                      const isOpen = !!openHistoryRows[entry.reportId]
                      const verificationState = verificationByReportId[entry.reportId] || null
                      const isVerifying = verificationState?.status === 'loading'

                      return (
                        <div key={key} className="history-item">
                          <div className="history-row">
                            <span className="history-time">{entry.timestamp || '—'}</span>
                            <span className="history-id">{entry.reportId || '—'}</span>
                            <button
                              type="button"
                              className="history-dropdown"
                              onClick={() => toggleHistoryRow(entry.reportId)}
                              aria-label={isOpen ? 'Collapse metadata' : 'Expand metadata'}
                              aria-expanded={isOpen}
                            >
                              {isOpen ? '▾' : '▸'}
                            </button>
                          </div>

                          {isOpen && (
                            <div className="history-meta">
                              <ul className="chain-list">
                                <li>
                                  <span>Report ID</span>
                                  <strong>{entry.reportId || '—'}</strong>
                                </li>
                                <li>
                                  <span>Timestamp</span>
                                  <strong>{entry.timestamp || '—'}</strong>
                                </li>
                                <li>
                                  <span>Model</span>
                                  <strong>
                                    {(entry.model?.name || '—')} ({entry.model?.version || '—'})
                                  </strong>
                                </li>
                                <li>
                                  <span>JSON CID</span>
                                  {entry.json_cid ? (
                                    <a href={IPFS_GATEWAY_BASE + entry.json_cid} target="_blank" rel="noreferrer">
                                      {entry.json_cid}
                                    </a>
                                  ) : (
                                    <strong>—</strong>
                                  )}
                                </li>
                                <li>
                                  <span>PDF CID</span>
                                  {entry.pdf_cid ? (
                                    <a href={IPFS_GATEWAY_BASE + entry.pdf_cid} target="_blank" rel="noreferrer">
                                      {entry.pdf_cid}
                                    </a>
                                  ) : (
                                    <strong>—</strong>
                                  )}
                                </li>
                                <li>
                                  <span>JSON Hash</span>
                                  <strong className="mono">{entry.hash || '—'}</strong>
                                </li>
                                <li>
                                  <span>Tx Hash</span>
                                  {entry.tx_hash ? (
                                    <a href={SEPOLIA_TX_BASE + entry.tx_hash} target="_blank" rel="noreferrer" className="mono">
                                      {entry.tx_hash}
                                    </a>
                                  ) : (
                                    <strong>—</strong>
                                  )}
                                </li>
                              </ul>

                              <div className="verify-row">
                                <button
                                  type="button"
                                  className="secondary-action verify-button"
                                  onClick={() => verifyHistoryReport(entry)}
                                  disabled={isVerifying || !entry.json_cid}
                                >
                                  {isVerifying ? 'Verifying...' : 'Verify'}
                                </button>
                                {!!verificationState?.message && (
                                  <p className={`verify-message ${verificationState.status || ''}`}>
                                    {verificationState.message}
                                  </p>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
              <div className="support-card">
                <h4>Quick insights</h4>
                <p className="muted">Based on your current upload</p>
                <div className="insight-grid">
                  <div className="insight-tile">
                    <strong>{result ? formatPercent(result.prediction.probability) : '—'}</strong>
                    <span>Top‑1 confidence</span>
                  </div>
                  <div className="insight-tile">
                    <strong>
                      {typeof result?.top5_combined === 'number' ? formatPercent(result.top5_combined) : '—'}
                    </strong>
                    <span>Top‑5 combined</span>
                  </div>
                  <div className="insight-tile">
                    <strong>{result?.used_model || '—'}</strong>
                    <span>Model route</span>
                  </div>
                  <div className="insight-tile">
                    <strong>{quality.blur === 'sharp' ? 'Clear' : quality.blur === 'blur' ? 'Blur' : '—'}</strong>
                    <span>Image clarity</span>
                  </div>
                </div>
              </div>
            </div>
          </form>

          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel results">
          <div className="panel-header">
            <h2>Assessment</h2>
          </div>

          {!result && (
            <div className="empty">
              <p>Results will appear here once an image is analyzed.</p>
            </div>
          )}

          {result && (
            <div className="result-body">
              <div className="primary">
                <div>
                  <h3>
                    <button
                      type="button"
                      className="disease-link"
                      onClick={() => setDiseaseInfoOpen(result.prediction.label)}
                    >
                      {formatLabel(result.prediction.label)}
                    </button>
                  </h3>
                  <p className="confidence">Confidence: {formatPercent(result.prediction.probability)}</p>
                  <div className="confidence-bar">
                    <span style={{ width: `${clamp(result.prediction.probability * 100, 3, 100)}%` }} />
                  </div>
                  {typeof result.top5_combined === 'number' && (
                    <p className="hint">Top‑5 combined confidence: {formatPercent(result.top5_combined)}</p>
                  )}
                </div>
                <div className="risk-chip">
                  {result.guidance.risk === 'high'
                    ? 'Needs clinical review'
                    : result.guidance.risk === 'moderate'
                      ? 'Inconclusive risk'
                      : 'Low confidence result'}
                </div>
              </div>

              <div className="grid">
                <div>
                  <h4>Top matches</h4>
                  <ul className="top-matches">
                    {result.top3.map((item) => (
                      <li key={item.index}>
                        <span>
                          <button
                            type="button"
                            className="disease-link inline"
                            onClick={() => setDiseaseInfoOpen(item.label)}
                          >
                            {formatLabel(item.label)}
                          </button>
                        </span>
                        <div className="match-row">
                          <div className="match-bar">
                            <span style={{ width: `${clamp(item.probability * 100, 2, 100)}%` }} />
                          </div>
                          <span className="match-percent">{formatPercent(item.probability)}</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                  <div className="micro-card">
                    <h5>How to read this</h5>
                    <p className="muted">Top‑5 accuracy is more reliable for many classes. Low confidence means “inconclusive.”</p>
                  </div>
                </div>

                <div>
                  <h4>Next steps</h4>
                  <ul>
                    {result.guidance.next_steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4>Wellness tips</h4>
                  <ul>
                    {result.guidance.wellness_tips.map((tip) => (
                      <li key={tip}>{tip}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="warnings">
                {result.guidance.warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>

              <div className="chain-card">
                <h4>Blockchain report metadata</h4>
                {result.persistence_status !== 'complete' ? (
                  <p className="processing-note">processing transaction...</p>
                ) : (
                  <p className="hint">Verification status is intentionally not stored locally.</p>
                )}
                <ul className="chain-list">
                  <li>
                    <span>Report ID</span>
                    <strong>{result.reportId || '—'}</strong>
                  </li>
                  <li>
                    <span>Timestamp</span>
                    <strong>{result.timestamp || '—'}</strong>
                  </li>
                  <li>
                    <span>Model</span>
                    <strong>
                      {(result.model?.name || result.used_model || '—')} ({result.model?.version || result.model_version || '—'})
                    </strong>
                  </li>
                  <li>
                    <span>JSON CID</span>
                    {result.json_cid ? (
                      <a href={IPFS_GATEWAY_BASE + result.json_cid} target="_blank" rel="noreferrer">
                        {result.json_cid}
                      </a>
                    ) : (
                      <strong>—</strong>
                    )}
                  </li>
                  <li>
                    <span>PDF CID</span>
                    {result.pdf_cid ? (
                      <a href={IPFS_GATEWAY_BASE + result.pdf_cid} target="_blank" rel="noreferrer">
                        {result.pdf_cid}
                      </a>
                    ) : (
                      <strong>—</strong>
                    )}
                  </li>
                  <li>
                    <span>JSON Hash</span>
                    <strong className="mono">{result.hash || '—'}</strong>
                  </li>
                  <li>
                    <span>Tx Hash</span>
                    {result.tx_hash ? (
                      <a href={SEPOLIA_TX_BASE + result.tx_hash} target="_blank" rel="noreferrer" className="mono">
                        {result.tx_hash}
                      </a>
                    ) : (
                      <strong>—</strong>
                    )}
                  </li>
                </ul>
              </div>

              <details>
                <summary>How this AI works</summary>
                <ul>
                  <li>Trained on dermatology image datasets (ISIC 2019 + SD-198).</li>
                  <li>Not a diagnostic tool; results are educational.</li>
                  <li>Accuracy varies with lighting, focus, and skin tone.</li>
                </ul>
              </details>
            </div>
          )}
        </section>
      </main>

      <section className="terms-panel" aria-label="Terms and conditions">
        <h2>Terms & Conditions</h2>
        <ul>
          <li>This app is for educational use only and does not provide medical diagnosis or treatment.</li>
          <li>Predictions may be incorrect due to image quality, skin tone variation, and model limitations.</li>
          <li>Do not delay or avoid professional care based on this tool&apos;s output.</li>
          <li>For urgent symptoms such as bleeding, severe pain, or rapid lesion changes, seek in-person care immediately.</li>
          <li>By using this app, you accept full responsibility for how results are interpreted and used.</li>
        </ul>
      </section>

      {diseaseInfoOpen && (
        <div className="modal-backdrop" onClick={() => setDiseaseInfoOpen(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4>{formatLabel(diseaseInfoOpen)}</h4>
              <button type="button" className="modal-close" onClick={() => setDiseaseInfoOpen(null)}>
                ×
              </button>
            </div>
            <p>{getDiseaseInfo(diseaseInfoOpen)}</p>
            <p className="muted">Educational note only — not a diagnosis.</p>
          </div>
        </div>
      )}
    </div>
  )
}
