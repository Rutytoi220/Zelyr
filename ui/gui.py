"""Simple tkinter GUI for ChienGPT / Zelyr."""

import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
from brain.agent import Agent
from utils.logger import get_logger
from utils.ollama import is_server_up, which_ollama, start_ollama

logger = get_logger(__name__)


class ChienGPTGUI:
    def __init__(self, root, model: str = None, system_prompt: str = None, verbose: bool = False):
        self.root = root
        self.root.title("ChienGPT - Local AI Assistant")
        self.root.geometry("900x700")
        
        self.agent = Agent(model=model, system_prompt=system_prompt)
        self.verbose = verbose
        self.is_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Build the UI layout."""
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(
            header_frame, 
            text="🤖 ChienGPT - Local AI Assistant",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(
            header_frame,
            text="● Checking...",
            foreground="orange"
        )
        self.status_label.pack(side=tk.RIGHT)
        self._update_server_status()
        
        # Model selection
        model_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=self.agent.model)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=20)
        model_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_frame, text="Apply", command=self._apply_model).pack(side=tk.LEFT, padx=5)
        
        # Output area
        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=15,
            width=80,
            state=tk.DISABLED,
            font=("Courier", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.output_text.tag_config("info", foreground="#4ec9b0")
        self.output_text.tag_config("error", foreground="#f48771")
        self.output_text.tag_config("success", foreground="#6a9955")
        self.output_text.tag_config("user", foreground="#9cdcfe", font=("Courier", 10, "bold"))
        self.output_text.tag_config("ai", foreground="#ce9178")
        
        # Print welcome message
        self._log_info("Welcome to ChienGPT! Type your commands below or use buttons for quick actions.")
        self._log_info("Type 'help' for available commands.\n")
        
        # Input area
        input_frame = ttk.LabelFrame(self.root, text="Command Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.input_text = tk.Text(input_frame, height=3, width=80, font=("Courier", 10))
        self.input_text.pack(fill=tk.X)
        self.input_text.bind("<Return>", self._handle_input_key)
        self.input_text.bind("<Shift-Return>", self._handle_input_key)
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="📤 Send (Enter)",
            command=self._send_command
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🔍 Vision Mode",
            command=self._vision_mode
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="⚙️ Ollama Status",
            command=self._ollama_status
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🚀 Start Ollama",
            command=self._start_ollama
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="📋 Help",
            command=self._show_help
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self._clear_output
        ).pack(side=tk.LEFT, padx=5)
    
    def _log_info(self, msg: str):
        """Log an info message to output."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg + "\n", "info")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _log_user(self, msg: str):
        """Log a user message."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"💬 You: {msg}\n", "user")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _log_ai(self, msg: str):
        """Log an AI response."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"🤖 AI: {msg}\n", "ai")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _log_success(self, msg: str):
        """Log a success message."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg + "\n", "success")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _log_error(self, msg: str):
        """Log an error message."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, msg + "\n", "error")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _send_command(self):
        """Send the input command to the agent."""
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Input", "Please enter a command.")
            return
        
        if self.is_processing:
            messagebox.showwarning("Processing", "Please wait for the current command to finish.")
            return
        
        self._log_user(prompt)
        self.input_text.delete("1.0", tk.END)
        
        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=self._process_command, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def _handle_input_key(self, event):
        """Handle Return and Shift+Return in input text."""
        if event.state & 0x1:  # Shift is pressed
            # Insert newline for Shift+Return
            self.input_text.insert(tk.INSERT, "\n")
            return "break"
        else:
            # Send command for plain Return
            self._send_command()
            return "break"
    
    def _process_command(self, prompt: str):
        """Process command in background thread."""
        self.is_processing = True
        try:
            result = self.agent.handle_prompt(prompt)
            
            # Extract the nested result dict
            cmd_result = result.get('result', {})
            is_ok = cmd_result.get('ok', False)
            
            if is_ok:
                msg = cmd_result.get('message', 'Command executed')
                self._log_success(f"✅ {msg}")
            else:
                msg = cmd_result.get('message', 'Unknown error')
                # Make sure we always have a message to display
                if not msg or msg.strip() == '':
                    msg = 'Command failed with no error details'
                self._log_error(f"❌ Error: {msg}")
            
            # Show parsed instruction if verbose
            if self.verbose and result.get('parsed'):
                self._log_info(f"[DEBUG] Parsed: {result['parsed']}")
        except Exception as e:
            self._log_error(f"❌ Error: {str(e)}")
            logger.exception("Error processing command")
        finally:
            self.is_processing = False
    
    def _vision_mode(self):
        """Open vision mode dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Vision Mode")
        dialog.geometry("500x200")
        
        ttk.Label(dialog, text="Enter the task for vision mode:", font=("Helvetica", 10)).pack(padx=10, pady=10)
        
        task_text = tk.Text(dialog, height=4, width=50, font=("Courier", 10))
        task_text.pack(padx=10, pady=5)
        
        def execute_vision():
            task = task_text.get("1.0", tk.END).strip()
            if not task:
                messagebox.showwarning("Empty Task", "Please enter a vision task.")
                return
            
            if self.is_processing:
                messagebox.showwarning("Processing", "Please wait for current operation to finish.")
                return
            
            dialog.destroy()
            self._log_user(f":vision {task}")
            thread = threading.Thread(target=self._execute_vision_task, args=(task,))
            thread.daemon = True
            thread.start()
        
        ttk.Button(dialog, text="Execute Vision Task", command=execute_vision).pack(pady=10)
        ttk.Label(dialog, text="Examples:\n• Click the Firefox icon\n• Open a web browser\n• Take a screenshot", 
                 font=("Helvetica", 9), justify=tk.LEFT).pack(padx=10, pady=5)
    
    def _execute_vision_task(self, task: str):
        """Execute a vision task in background."""
        self.is_processing = True
        try:
            self._log_info("\n🔍 Starting vision task...")
            self._log_info("Taking screenshots and analyzing desktop...\n")
            
            result = self.agent.handle_vision_task(task, max_steps=10)
            
            if result.get('ok'):
                self._log_success(f"✅ Vision task completed!")
                self._log_success(f"   Steps executed: {result.get('steps', 0)}")
                msg = result.get('message', 'Done')
                self._log_success(f"   Result: {msg}")
            else:
                self._log_error(f"❌ Vision task failed")
                msg = result.get('message', 'Unknown error')
                self._log_error(f"   Error: {msg}")
        except Exception as e:
            self._log_error(f"❌ Vision error: {str(e)}")
            logger.exception("Vision task error")
        finally:
            self.is_processing = False
    
    def _ollama_status(self):
        """Show Ollama server status."""
        try:
            up = is_server_up()
            binp = which_ollama()
            
            self._log_info("\n⚙️ Ollama Status:")
            self._log_success(f"   Server reachable: {'✅ Yes' if up else '❌ No'}")
            self._log_info(f"   Binary on PATH: {'✅ Yes' if binp else '❌ No'}")
        except Exception as e:
            self._log_error(f"❌ Error checking Ollama: {str(e)}")
    
    def _start_ollama(self):
        """Start Ollama server."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Please wait for current operation to finish.")
            return
        
        thread = threading.Thread(target=self._start_ollama_thread)
        thread.daemon = True
        thread.start()
    
    def _start_ollama_thread(self):
        """Start Ollama in background thread."""
        self.is_processing = True
        try:
            binp = which_ollama()
            if not binp:
                self._log_error("❌ Ollama binary not found on PATH")
                self._log_info("   Install Ollama from: https://ollama.ai")
                return
            
            self._log_info("🚀 Starting Ollama...")
            ok, msg = start_ollama(binp)
            if ok:
                self._log_success(f"✅ {msg}")
                self._update_server_status()
            else:
                self._log_error(f"❌ {msg}")
        except Exception as e:
            self._log_error(f"❌ Error: {str(e)}")
            logger.exception("Ollama start error")
        finally:
            self.is_processing = False
    
    def _apply_model(self):
        """Apply the selected model."""
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showwarning("Invalid Model", "Please enter a model name.")
            return
        
        self.agent.model = model_name
        self._log_success(f"✅ Model changed to: {model_name}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
📚 Zelyr Commands:

TEXT COMMANDS:
• Regular queries: "What is Python?"
• Open apps: "open firefox", "open discord and spotify"
• Open folders: "open /path/to/folder"
• System commands: Any shell command

SPECIAL COMMANDS:
• :vision <task> - Use AI + vision to control desktop
  Examples:
  - :vision click the blue button
  - :vision open firefox and search for github
  - :vision take a screenshot and describe it

VISION MODE BUTTON:
• Click "🔍 Vision Mode" to enter interactive vision tasks
• Can execute multiple steps automatically

OLLAMA MANAGEMENT:
• ⚙️ Ollama Status - Check server connection
• 🚀 Start Ollama - Attempt to start daemon

SETTINGS:
• Change model name in dropdown
• Ctrl+Enter to send commands quickly

EXAMPLES:
💡 "who won the world cup?"
💡 "open firefox"
💡 ":vision click on the google search bar"
💡 "tell me a joke"
"""
        messagebox.showinfo("Help", help_text)
    
    def _clear_output(self):
        """Clear the output area."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def _update_server_status(self):
        """Update Ollama server status indicator."""
        def check_status():
            try:
                up = is_server_up()
                status_text = "● Server Online" if up else "● Server Offline"
                color = "green" if up else "red"
                self.status_label.config(text=status_text, foreground=color)
            except Exception as e:
                self.status_label.config(text="● Unknown", foreground="gray")
        
        thread = threading.Thread(target=check_status)
        thread.daemon = True
        thread.start()


def run_gui(model: str = None, verbose: bool = False, system_prompt: str = None):
    """Launch the GUI application."""
    root = tk.Tk()
    app = ChienGPTGUI(root, model=model, system_prompt=system_prompt, verbose=verbose)
    root.mainloop()
