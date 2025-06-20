"""
Build Your Own Shell – Minimal Python Shell Interface
Author: Ihunna Amugo | Build-Your-Own-X Series

🖥️ Goal:
Implement a basic command-line shell that:
- Shows a prompt
- Runs basic system commands (ls, echo, etc.)
- Handles 'cd' and 'exit' internally
- Uses subprocess to execute external commands

🔁 Real-World Connection:
This is how SLURM job scripts are launched in HPC (think: `sbatch`, `srun`, etc).
A custom shell gives full control over simulation pipelines.
"""

import os
import subprocess

def ihunna_shell():
    while True:
        # Show current working directory as part of the prompt
        cwd = os.getcwd()
        command = input(f"ihunna-shell:{cwd}$ ").strip()

        # Handle empty input
        if command == "":
            continue

        # Built-in commands: exit
        if command.lower() in ["exit", "quit"]:
            print("🛑 Exiting Ihunna Shell.")
            break

        # Built-in command: change directory
        if command.startswith("cd "):
            path = command[3:].strip()
            try:
                os.chdir(path)
            except Exception as e:
                print(f"cd: {e}")
            continue

        # Use subprocess to run everything else
        try:
            result = subprocess.run(command, shell=True)
        except Exception as e:
            print(f"Command error: {e}")

if __name__ == "__main__":
    ihunna_shell()
