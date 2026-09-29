import { useState } from 'react';
import './index.css';

function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const protocols = [
    { id: 'pd', name: "Parkinson's Disease (PD)" },
    { id: 'ms', name: "Multiple Sclerosis (MS)" },
    { id: 'hd', name: "Huntington's Disease (HD)" }
  ];
  const [selectedProtocol, setSelectedProtocol] = useState(protocols[0]);
  const [files, setFiles] = useState<Record<number, File>>({});

  const handleFileChange = (index: number, file: File) => {
    setFiles(prev => ({ ...prev, [index]: file }));
  };

  const handleDiagnose = async () => {
    setAnalyzing(true);
    setResults(null);
    
    try {
      const formData = new FormData();
      formData.append('protocol', selectedProtocol.id);
      
      // Order maps to the UI grid: [Imaging, Speech, Gait Sensor, Genetics, Handwriting, Clinical]
      const fileKeys = ['img_file', 'speech_file', 'sensor_file', 'genetic_file', 'hw_file', 'clinical_file'];
      Object.keys(files).forEach((key) => {
         const i = parseInt(key);
         if (files[i]) {
            formData.append(fileKeys[i], files[i]);
         }
      });
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
         throw new Error("Server error");
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the Neural Backend API.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="dashboard-container">
      <div className="background-orbs">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
      </div>

      <header className="glass-header">
        <div className="logo-container">
          <div className="brain-icon"></div>
          <h1>NeuroDetect AI</h1>
        </div>
        <div className="user-profile">Dr. Alex Chen</div>
      </header>

      <main className="main-content">
        <div className="glass-panel upload-section">
          <h2>Patient Data Pipeline</h2>
          <p>Upload multi-modal inputs for unified analysis.</p>

          <div className="protocol-selector" style={{ marginBottom: '25px', marginTop: '15px' }}>
            <label style={{ marginRight: '12px', fontWeight: 'bold' }}>Target Screening Protocol:</label>
            <select 
              value={selectedProtocol.id} 
              onChange={(e) => setSelectedProtocol(protocols.find(p => p.id === e.target.value) || protocols[0])}
              style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(0,0,0,0.3)', color: 'white', outline: 'none', cursor: 'pointer' }}
            >
              {protocols.map(p => (
                <option key={p.id} value={p.id} style={{ color: 'black' }}>{p.name}</option>
              ))}
            </select>
          </div>
          
          <div className="modality-grid">
            {['Imaging (MRI)', 'Speech (.wav)', 'Gait Sensor', 'Genetics (.csv)', 'Handwriting', 'Clinical Data'].map((mod, i) => (
              <div className="modality-card" key={i}>
                <div className="status-indicator" style={files[i] ? { backgroundColor: '#4ade80', boxShadow: '0 0 10px #4ade80' } : {}}></div>
                <h3>{mod}</h3>
                <label className="upload-btn" style={{ cursor: 'pointer', display: 'inline-block', textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '100%' }}>
                  {files[i] ? files[i].name : 'Select File'}
                  <input 
                    type="file" 
                    style={{ display: 'none' }} 
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFileChange(i, e.target.files[0]);
                      }
                    }} 
                  />
                </label>
              </div>
            ))}
          </div>

          <button 
            className={`diagnose-btn ${analyzing ? 'pulsing' : ''}`} 
            onClick={handleDiagnose}
            disabled={analyzing}
          >
            {analyzing ? 'Synthesizing Modalities...' : 'Run Neural Diagnosis'}
          </button>
        </div>

        <div className="glass-panel results-section">
          <h2>Diagnosis & Explainability</h2>
          
          {!results && !analyzing && (
            <div className="empty-state">
              <p>Awaiting patient data synthesis.</p>
            </div>
          )}

          {analyzing && (
            <div className="analyzing-state">
              <div className="loader"></div>
              <p>Fusing 6 Modalities...</p>
            </div>
          )}

          {results && (
            <div className="results-display fade-in">
              <div className="primary-diagnosis">
                <h3>Prediction</h3>
                <div className="diagnosis-text">{results.diagnosis}</div>
                <div className="confidence">Confidence: {results.confidence}%</div>
              </div>

              <div className="explainability">
                <h3>Modality Contributions (Attention Weights)</h3>
                <div className="chart-container">
                  {results.contributions.map((c: any, i: number) => (
                    <div className="bar-row" key={i}>
                      <span className="label">{c.modality}</span>
                      <div className="bar-track">
                        <div 
                          className="bar-fill" 
                          style={{ width: `${c.value}%`, animationDelay: `${i * 0.1}s` }}
                        ></div>
                      </div>
                      <span className="value">{c.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
