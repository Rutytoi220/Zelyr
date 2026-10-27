import React, { useEffect, useRef, useState } from 'react';
import { Terminal } from '@xterm/xterm';
import '@xterm/xterm/css/xterm.css';

interface TerminalMirrorProps {
  wsUrl: string;
}

const TerminalMirror: React.FC<TerminalMirrorProps> = ({ wsUrl }) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<Terminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [inputValue, setInputValue] = useState('');

  useEffect(() => {
    if (!terminalRef.current) return;

    const term = new Terminal({
      theme: {
        background: 'transparent',
        foreground: '#e5e7eb',
        cursor: '#00ffcc',
        selectionBackground: 'rgba(0, 255, 204, 0.3)',
      },
      fontFamily: "'Fira Code', 'Courier New', monospace",
      fontSize: 14,
      cursorBlink: true,
      disableStdin: true,
      convertEol: true,
    });

    term.open(terminalRef.current);
    xtermRef.current = term;

    term.writeln('\x1b[36m[System]\x1b[0m Initializing Zelyr Terminal Mirror...');

    let reconnectInterval: ReturnType<typeof setInterval>;

    const connectWebSocket = () => {
      term.writeln(`\x1b[33m[System]\x1b[0m Connecting to gateway at ${wsUrl}...`);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        term.writeln('\x1b[32m[System]\x1b[0m WebSocket connection established. Authenticating...');
        ws.send(JSON.stringify({
          method: "zelyr.authenticate",
          token: import.meta.env.VITE_DAEMON_TOKEN 
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'log') {
            term.writeln(data.content);
          } else if (data.event) {
             term.writeln(`\x1b[90m[Event]\x1b[0m ${data.event}`);
          } else {
             term.writeln(event.data);
          }
        } catch (e) {
          term.writeln(event.data);
        }
      };

      ws.onclose = () => {
        term.writeln('\x1b[31m[System]\x1b[0m WebSocket connection closed. Reconnecting in 3s...');
        clearInterval(reconnectInterval);
        reconnectInterval = setInterval(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        term.writeln('\x1b[31m[Error]\x1b[0m WebSocket encountered an error.');
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      clearInterval(reconnectInterval);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      term.dispose();
    };
  }, [wsUrl]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim() !== '') {
      const text = inputValue.trim();
      
      // Send JSON-RPC payload
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const payload = {
          method: 'zelyr.prompt',
          params: { text: text }
        };
        wsRef.current.send(JSON.stringify(payload));
      }
      
      // Echo to local terminal
      if (xtermRef.current) {
        xtermRef.current.writeln(`\x1b[32m> user:\x1b[0m ${text}`);
      }
      
      setInputValue('');
    }
  };

  return (
    <div className="flex flex-col flex-1 w-full h-full bg-black/40 backdrop-blur-md border border-zelyr-border rounded-lg overflow-hidden shadow-2xl">
      <div className="flex-1 w-full p-2" ref={terminalRef} />
      <div className="flex items-center px-4 py-3 bg-black/60 border-t border-zelyr-border">
        <span className="text-zelyr-accent mr-3 font-mono text-sm">{'>'}</span>
        <input 
          type="text" 
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter command or prompt..." 
          className="flex-1 bg-transparent border-none outline-none text-zelyr-text font-mono text-sm placeholder-zelyr-muted"
          autoFocus
        />
      </div>
    </div>
  );
};

export default TerminalMirror;
