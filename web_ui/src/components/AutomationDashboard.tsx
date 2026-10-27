import React, { useEffect, useState } from 'react';
import { Play, Square, Trash2, Video, Activity, ListOrdered, CheckCircle2 } from 'lucide-react';

interface AutomationDashboardProps {
  wsUrl: string;
}

interface Macro {
  id: string;
  name: string;
  created_at: string;
  steps: any[];
}

const AutomationDashboard: React.FC<AutomationDashboardProps> = ({ wsUrl }) => {
  const [macros, setMacros] = useState<Macro[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [newMacroName, setNewMacroName] = useState('New Macro');
  const [liveSteps, setLiveSteps] = useState<any[]>([]);
  
  // Replay State
  const [activeMacroId, setActiveMacroId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(0);
  const [lastAction, setLastAction] = useState<string>("");

  const fetchMacros = async () => {
    try {
      const res = await fetch('http://127.0.0.1:49103/api/macros');
      if (res.ok) {
        const data = await res.json();
        setMacros(data.macros || []);
      }
    } catch (e) {
      console.error("Failed to fetch macros", e);
    }
  };

  useEffect(() => {
    fetchMacros();
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectInterval: ReturnType<typeof setInterval>;

    const connect = () => {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        ws.send(JSON.stringify({
          method: "zelyr.authenticate",
          token: import.meta.env.VITE_DAEMON_TOKEN 
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.method === 'zelyr.event' && data.params && data.params.event_type) {
            const eventType = data.params.event_type;
            const payload = data.params.payload || {};
            
            if (eventType === 'tool.executed' && isRecording) {
              setLiveSteps(prev => [...prev, payload]);
            } else if (eventType === 'macro.step') {
              setActiveMacroId(payload.macro_id);
              setCurrentStep(payload.step);
              setTotalSteps(payload.total);
              setLastAction(payload.action);
            } else if (eventType === 'macro.completed') {
              setTimeout(() => {
                setActiveMacroId(null);
                setCurrentStep(0);
                setTotalSteps(0);
                setLastAction("");
              }, 2000);
            }
          }
        } catch (e) {
          // Ignore
        }
      };

      ws.onclose = () => {
        clearInterval(reconnectInterval);
        reconnectInterval = setInterval(connect, 3000);
      };
    };

    connect();

    return () => {
      clearInterval(reconnectInterval);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [wsUrl, isRecording]);

  const handleStartRecording = async () => {
    try {
      await fetch('http://127.0.0.1:49103/api/macros/start', { method: 'POST' });
      setIsRecording(true);
      setLiveSteps([]);
    } catch (e) {
      console.error(e);
    }
  };

  const handleStopRecording = async () => {
    try {
      await fetch('http://127.0.0.1:49103/api/macros/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newMacroName })
      });
      setIsRecording(false);
      setNewMacroName('New Macro');
      setLiveSteps([]);
      fetchMacros();
    } catch (e) {
      console.error(e);
    }
  };

  const handleRunMacro = async (macroId: string) => {
    try {
      await fetch(`http://127.0.0.1:49103/api/macros/${macroId}/execute`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteMacro = async (macroId: string) => {
    try {
      await fetch(`http://127.0.0.1:49103/api/macros/${macroId}`, { method: 'DELETE' });
      fetchMacros();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="w-full bg-black/40 backdrop-blur-md border border-zelyr-border rounded-lg p-4 flex flex-col gap-4 shadow-xl mb-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-zelyr-text font-bold tracking-wider text-sm">
          <Activity className="w-4 h-4 text-zelyr-accent" />
          AUTOMATION CONTROL CENTER
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Macro Studio */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-md p-3 flex flex-col">
          <h3 className="text-xs font-mono text-zelyr-accent uppercase mb-3 flex items-center gap-2">
            <ListOrdered className="w-4 h-4" />
            Macro Studio
          </h3>
          <div className="flex-1 overflow-y-auto space-y-2 max-h-48 pr-2">
            {macros.length === 0 ? (
              <div className="text-zinc-500 text-xs italic font-mono">No macros saved yet.</div>
            ) : (
              macros.map(m => (
                <div key={m.id} className="bg-zinc-950/80 border border-zinc-800/50 rounded p-2 flex items-center justify-between group hover:border-zelyr-accent/30 transition-colors">
                  <div>
                    <div className="text-sm font-semibold text-zinc-200">{m.name}</div>
                    <div className="text-[10px] text-zinc-500 font-mono mt-0.5">{m.steps?.length || 0} steps</div>
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => handleRunMacro(m.id)} className="p-1.5 bg-zelyr-accent/10 text-zelyr-accent hover:bg-zelyr-accent/20 rounded">
                      <Play className="w-3 h-3" />
                    </button>
                    <button onClick={() => handleDeleteMacro(m.id)} className="p-1.5 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Live Recorder HUD & Replay Viewer */}
        <div className="flex flex-col gap-4">
          
          {/* Recorder */}
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-md p-3 flex flex-col">
            <h3 className="text-xs font-mono text-red-400 uppercase mb-3 flex items-center gap-2">
              <Video className="w-4 h-4" />
              Live Recorder HUD
            </h3>
            
            {!isRecording ? (
              <div className="flex flex-col gap-2">
                <input 
                  type="text" 
                  value={newMacroName}
                  onChange={e => setNewMacroName(e.target.value)}
                  className="bg-black/50 border border-zinc-700 text-sm p-1.5 rounded focus:border-red-400 focus:outline-none text-zinc-300 font-mono"
                  placeholder="Macro Name..."
                />
                <button 
                  onClick={handleStartRecording}
                  className="flex items-center justify-center gap-2 w-full bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 py-2 rounded text-sm font-semibold transition-colors uppercase tracking-wider"
                >
                  <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  Record New Macro
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <div className="text-xs text-red-400 animate-pulse font-mono mb-1">● RECORDING ACTIVE...</div>
                <div className="h-20 overflow-y-auto bg-black/60 border border-zinc-800 rounded p-2 text-[10px] font-mono text-zinc-400 space-y-1">
                  {liveSteps.length === 0 ? (
                    <span className="opacity-50">Waiting for actions...</span>
                  ) : (
                    liveSteps.map((step, idx) => (
                      <div key={idx}><span className="text-zelyr-accent">&gt;</span> {step.tool_name || step.tool}</div>
                    ))
                  )}
                </div>
                <button 
                  onClick={handleStopRecording}
                  className="flex items-center justify-center gap-2 w-full bg-zinc-800 hover:bg-zinc-700 text-white border border-zinc-600 py-2 rounded text-sm font-semibold transition-colors uppercase tracking-wider mt-1"
                >
                  <Square className="w-4 h-4" />
                  Stop & Save
                </button>
              </div>
            )}
          </div>

          {/* Replay Viewer */}
          {activeMacroId && (
            <div className="bg-zelyr-accent/5 border border-zelyr-accent/20 rounded-md p-3 flex flex-col animate-in fade-in slide-in-from-bottom-2">
              <h3 className="text-xs font-mono text-zelyr-accent uppercase mb-2 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Step-by-Step Replay Viewer
              </h3>
              
              <div className="flex items-center gap-3">
                <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-zelyr-accent transition-all duration-300"
                    style={{ width: `${(currentStep / totalSteps) * 100}%` }}
                  />
                </div>
                <div className="text-xs font-mono text-zelyr-accent font-bold">
                  {currentStep} / {totalSteps}
                </div>
              </div>
              <div className="text-[10px] font-mono text-zinc-400 mt-2 truncate">
                Executing: <span className="text-white">{lastAction}</span>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default AutomationDashboard;
