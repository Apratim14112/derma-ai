import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

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
  const videoRef = useRef(null)
  const streamRef = useRef(null)

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
        body: formData
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }
      if (preview) {
        setHistory((prev) => [preview, ...prev].slice(0, 2))
      }
      setResult(data)
    } catch (err) {
      setError(err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
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
            <button type="button" className="secondary-action" onClick={() => window.print()}>
              Save result as PDF
            </button>
          </div>
        </div>
        <div className="hero-card">
          <h3>Safety first</h3>
          <ul>
            <li>For education only — not medical advice.</li>
            <li>Urgent symptoms need in-person care.</li>
            <li>Protect your skin daily with SPF 30+.</li>
          </ul>
        </div>
      </header>

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

            <div>
              <p className="hint">Image quality checks (beta)</p>
              <ul className="quality-list">
                <li className={quality.face === 'yes' ? 'ok' : quality.face === 'no' ? 'warn' : ''}>
                  {quality.face === 'yes'
                    ? '✔ Face detected'
                    : quality.face === 'no'
                      ? '⚠ Face not detected'
                      : '• Face detection not available'}
                </li>
                <li className={quality.lighting === 'good' ? 'ok' : quality.lighting === 'low' ? 'warn' : ''}>
                  {quality.lighting === 'good'
                    ? '✔ Adequate lighting'
                    : quality.lighting === 'low'
                      ? '⚠ Low lighting'
                      : quality.lighting === 'high'
                        ? '⚠ Very bright lighting'
                        : '• Lighting check pending'}
                </li>
                <li className={quality.blur === 'sharp' ? 'ok' : quality.blur === 'blur' ? 'warn' : ''}>
                  {quality.blur === 'sharp'
                    ? '✔ Clear image'
                    : quality.blur === 'blur'
                      ? '⚠ Slight blur detected'
                      : '• Blur check pending'}
                </li>
              </ul>
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
                    <strong>{history.length}</strong>
                    <span>saved images</span>
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
            <span>Model output + wellness guidance</span>
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
                  <div className="micro-card">
                    <h5>Next upload</h5>
                    <p className="muted">Re‑capture in similar lighting to compare changes.</p>
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
                  <h4>OTC suggestions</h4>
                  <div className="disclaimer-badge">General wellness guidance — not a prescription</div>
                  {result.guidance.otc_suggestions.length ? (
                    <>
                      <div className="otc-group">
                        <h5>Daily care</h5>
                        <ul>
                          {result.guidance.otc_suggestions
                            .filter((item) =>
                              /sunscreen|moisturizer|cleanser/i.test(item)
                            )
                            .map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                        </ul>
                      </div>
                      <div className="otc-group">
                        <h5>Relief</h5>
                        <ul>
                          {result.guidance.otc_suggestions
                            .filter((item) => /compress|petrolatum/i.test(item))
                            .map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                        </ul>
                      </div>
                      <div className="otc-group">
                        <h5>Short-term</h5>
                        <ul>
                          {result.guidance.otc_suggestions
                            .filter((item) => /hydrocortisone|antihistamine/i.test(item))
                            .map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                        </ul>
                      </div>
                    </>
                  ) : (
                    <p>None recommended. Seek clinician guidance.</p>
                  )}
                </div>

                <div>
                  <h4>Wellness tips</h4>
                  <ul>
                    {result.guidance.wellness_tips.map((tip) => (
                      <li key={tip}>{tip}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4>Image type</h4>
                  <ul>
                    <li>
                      <span>
                        Auto-detect: {result.modality?.label || 'unknown'}
                        {typeof result.modality?.confidence === 'number'
                          ? ` (${formatPercent(result.modality.confidence)})`
                          : ''}
                      </span>
                      <strong>Route: {result.used_model}</strong>
                    </li>
                    {result.acne_stage?.label && result.acne_stage.label !== 'n/a' && (
                      <li>
                        <span>
                          Stage‑1: {result.acne_stage.label.replace('_', ' ')}
                          {typeof result.acne_stage.confidence === 'number'
                            ? ` (${formatPercent(result.acne_stage.confidence)})`
                            : ''}
                        </span>
                        <strong>Decision</strong>
                      </li>
                    )}
                  </ul>
                </div>
              </div>

              <div className="warnings">
                {result.guidance.warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>

              <div className="trackers-grid">
                <div className="care-card">
                  <h4>Skin care tracker</h4>
                  <p className="hint">Simple daily checklist (local only)</p>
                  <label className="check-row">
                    <input type="checkbox" /> Gentle cleanse (AM/PM)
                  </label>
                  <label className="check-row">
                    <input type="checkbox" /> Moisturizer applied
                  </label>
                  <label className="check-row">
                    <input type="checkbox" /> SPF 30+ applied
                  </label>
                  <div className="chip-row">
                    <span className="chip">AM</span>
                    <span className="chip">PM</span>
                    <span className="chip">Sensitive</span>
                  </div>
                </div>
                <div className="care-card">
                  <h4>Medication log</h4>
                  <p className="hint">Track OTC use and reactions</p>
                  <div className="log-row">
                    <span>Hydrocortisone 1%</span>
                    <span className="muted">2 days</span>
                  </div>
                  <div className="log-row">
                    <span>Moisturizer</span>
                    <span className="muted">daily</span>
                  </div>
                  <div className="log-row">
                    <span>Sunscreen</span>
                    <span className="muted">daily</span>
                  </div>
                </div>
                <div className="care-card">
                  <h4>Symptom timeline</h4>
                  <div className="timeline">
                    <div className="timeline-row">
                      <span className="dot" />
                      <div>
                        <strong>Today</strong>
                        <p className="muted">Redness + papules</p>
                      </div>
                    </div>
                    <div className="timeline-row">
                      <span className="dot" />
                      <div>
                        <strong>2 weeks ago</strong>
                        <p className="muted">Mild irritation</p>
                      </div>
                    </div>
                    <div className="timeline-row">
                      <span className="dot" />
                      <div>
                        <strong>1 month ago</strong>
                        <p className="muted">No symptoms</p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="care-card">
                  <h4>Photo tips</h4>
                  <p className="hint">Keep angle and lighting consistent</p>
                  <div className="photo-row">
                    <div className="photo-thumb good">Even light</div>
                    <div className="photo-thumb bad">Harsh glare</div>
                  </div>
                  <div className="chip-row">
                    <span className="chip">No flash</span>
                    <span className="chip">Same distance</span>
                  </div>
                </div>
              </div>

              <div className="care-grid">
                <div className="care-card">
                  <h4>When to seek care</h4>
                  <p className="hint">Check any that apply. If yes, consider a clinician visit.</p>
                  <label className="check-row">
                    <input type="checkbox" /> Rapid change in size or color
                  </label>
                  <label className="check-row">
                    <input type="checkbox" /> Bleeding, crusting, or ulceration
                  </label>
                  <label className="check-row">
                    <input type="checkbox" /> Persistent pain or itching
                  </label>
                  <label className="check-row">
                    <input type="checkbox" /> Irregular borders or asymmetry
                  </label>
                </div>
                <div className="care-card">
                  <h4>Follow-up plan</h4>
                  <ul>
                    <li>Recheck with a clearer photo in 2–4 weeks.</li>
                    <li>Track changes in size, color, or texture.</li>
                    <li>Use the same lighting for comparisons.</li>
                  </ul>
                  <div className="section-note">Tip: Save this result as PDF for your records.</div>
                </div>
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
