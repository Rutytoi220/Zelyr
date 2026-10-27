import React, { useEffect, useState } from 'react';
import { Network, Brain, Code2, Scale, X, ScrollText } from 'lucide-react';

interface SwarmVisualizerProps {
  wsUrl: string;
}

type NodeStatus = 'idle' | 'active' | 'pass' | 'reject';

interface SwarmState {
  architect: NodeStatus;
  coder: NodeStatus;
  critic: NodeStatus;
}

const getStatusClasses = (status: NodeStatus) => {
  switch (status) {
    case 'idle':
      return 'bg-zinc-800 border-zinc-600 text-zinc-400';
    case 'active':
      return 'bg-zelyr-accent/20 border-zelyr-accent text-zelyr-accent animate-pulse shadow-[0_0_15px_rgba(0,255,204,0.4)]';
    case 'pass':
      return 'bg-green-500/20 border-green-500 text-green-400 shadow-[0_0_15px_rgba(34,197,94,0.4)]';
    case 'reject':
      return 'bg-red-500/20 border-red-500 text-red-400 shadow-[0_0_15px_rgba(239,68,68,0.4)]';
  }
};

const SwarmVisualizer: React.FC<SwarmVisualizerProps> = ({ wsUrl }) => {
  const [swarm, setSwarm] = useState<SwarmState>({
    architect: 'idle',
    coder: 'idle',
    critic: 'idle',
  });
  const [logs, setLogs] = useState<{persona: string, thought: string}[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activePersona, setActivePersona] = useState<string | null>(null);

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
          // Look for JSON-RPC method "zelyr.event" with params.event_type
          if (data.method === 'zelyr.event' && data.params && data.params.event_type) {
            const eventType = data.params.event_type;
            
            if (eventType.startsWith('swarm.')) {
              if (eventType === 'swarm.architect.thinking') setSwarm(s => ({ ...s, architect: 'active' }));
              else if (eventType === 'swarm.coder.patching') setSwarm(s => ({ ...s, coder: 'active' }));
              else if (eventType === 'swarm.critic.voting') setSwarm(s => ({ ...s, critic: 'active' }));
              else if (eventType === 'swarm.critic.pass') setSwarm(s => ({ ...s, critic: 'pass' }));
              else if (eventType === 'swarm.critic.reject') setSwarm(s => ({ ...s, critic: 'reject' }));
              else if (eventType === 'swarm.idle') setSwarm({ architect: 'idle', coder: 'idle', critic: 'idle' });
              else if (eventType === 'swarm.log') {
                if (data.params.payload && data.params.payload.persona) {
                  setLogs(prev => [...prev.slice(-49), { 
                    persona: data.params.payload.persona, 
                    thought: data.params.payload.thought 
                  }]);
                }
              }
            }
          }
        } catch (e) {
          // ignore parsing errors
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
  }, [wsUrl]);

  return (
    <div className="w-full bg-black/40 backdrop-blur-md border border-zelyr-border rounded-lg p-4 flex flex-col gap-3 shadow-xl mb-4">
      <div className="flex items-center gap-2 text-zelyr-text font-bold tracking-wider text-sm">
        <Network className="w-4 h-4 text-zelyr-accent" />
        SWARM CONSENSUS ENGINE
      </div>
      
      <div className="flex items-center justify-between gap-4 mt-2">
        {/* Architect Node */}
        <div 
          onClick={() => { setActivePersona('architect'); setDrawerOpen(true); }}
          className={`cursor-pointer flex-1 flex flex-col items-center justify-center p-3 rounded-lg border transition-all duration-300 ${getStatusClasses(swarm.architect)}`}
        >
          <Brain className="w-6 h-6 mb-2" />
          <span className="text-xs font-mono uppercase font-semibold">Architect</span>
          <span className="text-[10px] opacity-80 mt-1">{swarm.architect === 'active' ? 'Planning' : 'Idle'}</span>
        </div>

        {/* Coder Node */}
        <div 
          onClick={() => { setActivePersona('coder'); setDrawerOpen(true); }}
          className={`cursor-pointer flex-1 flex flex-col items-center justify-center p-3 rounded-lg border transition-all duration-300 ${getStatusClasses(swarm.coder)}`}
        >
          <Code2 className="w-6 h-6 mb-2" />
          <span className="text-xs font-mono uppercase font-semibold">Coder</span>
          <span className="text-[10px] opacity-80 mt-1">{swarm.coder === 'active' ? 'Patching' : 'Idle'}</span>
        </div>

        {/* Critic Node */}
        <div 
          onClick={() => { setActivePersona('critic'); setDrawerOpen(true); }}
          className={`cursor-pointer flex-1 flex flex-col items-center justify-center p-3 rounded-lg border transition-all duration-300 ${getStatusClasses(swarm.critic)}`}
        >
          <Scale className="w-6 h-6 mb-2" />
          <span className="text-xs font-mono uppercase font-semibold">Critic</span>
          <span className="text-[10px] opacity-80 mt-1">
            {swarm.critic === 'active' ? 'Voting' : swarm.critic === 'pass' ? 'Approved' : swarm.critic === 'reject' ? 'Rejected' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Consensus Log Drawer */}
      {drawerOpen && (
        <div className="mt-4 p-3 bg-zinc-900/80 border border-zelyr-accent/40 rounded-lg max-h-48 flex flex-col relative animate-in slide-in-from-top-2">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-zelyr-accent text-xs font-mono font-bold uppercase">
              <ScrollText className="w-4 h-4" />
              {activePersona} Chain of Thought
            </div>
            <button onClick={() => setDrawerOpen(false)} className="text-zinc-400 hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 text-xs font-mono text-zinc-300 pr-2">
            {logs.filter(l => l.persona === activePersona).length === 0 ? (
              <div className="text-zinc-500 italic">No logs recorded yet...</div>
            ) : (
              logs.filter(l => l.persona === activePersona).map((log, i) => (
                <div key={i} className="pb-1 border-b border-zinc-800/50">
                  <span className="text-zelyr-accent/70 mr-2">&gt;</span>{log.thought}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SwarmVisualizer;
