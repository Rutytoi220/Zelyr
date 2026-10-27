"""Simple CLI front-end for Zelyr using the Agent pipeline.

Adds `:ollama` commands to manage local Ollama installation and server:
- `:ollama status` — show whether server is reachable and whether binary exists
- `:ollama start` — attempt to start the Ollama daemon if binary is available
- `:ollama install` — show install instructions
- `:ollama install confirm` — attempt automated install (user consents)
"""

from brain.agent import Agent
from utils.logger import get_logger
from utils.ollama import is_server_up, which_ollama, start_ollama, attempt_auto_install

logger = get_logger(__name__)


def _handle_ollama_command(cmd_parts):
    verb = cmd_parts[1] if len(cmd_parts) > 1 else 'status'
    if verb == 'status':
        up = is_server_up()
        binp = which_ollama()
        print(f"Ollama server reachable: {up}")
        print(f"ollama binary on PATH: {binp or 'not found'}")
        return
    if verb == 'start':
        binp = which_ollama()
        if not binp:
            print('ollama binary not found. Use `:ollama install` to see installation instructions.')
            return
        ok, msg = start_ollama(binp)
        print(msg)
        return
    if verb == 'install':
        confirm = len(cmd_parts) > 2 and cmd_parts[2] == 'confirm'
        ok, msg = attempt_auto_install(confirm=confirm)
        print(msg)
        return
    print('Unknown :ollama command. Available: status, start, install')


def run_repl(model: str = None, verbose: bool = False, system_prompt: str = None):
    agent = Agent(model=model, system_prompt=system_prompt)
    print("\n" + "="*60)
    print("  Zelyr - Local AI Assistant")
    print("="*60)
    print("  Type 'help' for commands, 'exit' to quit\n")
    
    while True:
        try:
            prompt = input("💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋\n")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("\nGoodbye! 👋\n")
            break
        
        # Help command
        if prompt.lower() in ("help", "?"):
            print("\n📚 Available commands:")
            print("  - Open apps: 'open firefox', 'ouvre discord et opera'")
            print("  - Open folders: 'open /path/to/folder'")
            print("  - Run commands: Any system command")
            print("  - Vision mode: ':vision <task>' - Control PC with vision+LLM")
            print("  - :ollama status/start/install - Manage Ollama")
            print("  - exit/quit - Close Zelyr\n")
            print("💡 Vision examples:")
            print("  :vision click the blue button in the top right")
            print("  :vision open a web browser and search for github")
            print("  :vision take a screenshot and describe it\n")
            continue

        # Vision mode
        if prompt.startswith(':vision'):
            task = prompt[7:].strip()
            if not task:
                print("❌ Please provide a task for vision mode")
                print("Example: :vision click the Firefox icon\n")
                continue
            
            print(f"\n🔍 Starting vision task: {task}")
            print("Taking screenshots and analyzing desktop...\n")
            result = agent.handle_vision_task(task, max_steps=10)
            
            if result.get('ok'):
                print(f"✅ Vision task completed!")
                print(f"   Steps executed: {result.get('steps', 0)}")
                msg = result.get('message', 'Done')
                print(f"   Result: {msg}")
            else:
                print(f"❌ Vision task failed")
                msg = result.get('message', 'Unknown error')
                print(f"   Error: {msg}")
            print()
            continue

        # local management commands
        if prompt.startswith(':ollama'):
            parts = prompt.split()
            try:
                _handle_ollama_command(parts)
            except Exception as e:
                logger.exception('Failed handling :ollama command')
                print('❌ Error:', e)
            print()
            continue

        resp = agent.handle_prompt(prompt)
        parsed = resp.get('parsed', {})
        result = resp.get('result', {})
        
        # Display results with better formatting
        if parsed.get('type') == 'instruction':
            action = parsed.get('action')
            ok = result.get('ok')
            msg = result.get('message')
            
            if ok:
                print(f"✅ {msg}")
            else:
                print(f"❌ {msg}")
                
        elif parsed.get('type') == 'message':
            msg = result.get('message', '')
            print(f"\n🤖 Zelyr:\n{msg}\n")
            
        elif parsed.get('type') == 'no_action':
            print("ℹ️  No action needed.")
        else:
            print(f"⚠️  {result.get('message', 'Unknown response')}")
        
        print()
