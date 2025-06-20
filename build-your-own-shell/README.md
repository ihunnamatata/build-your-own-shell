# 🖥️ Build Your Own Shell (Python Mini-Shell)

This project is a simplified command-line shell interface written in Python. It mimics basic terminal behavior, supports navigation, and runs external system commands using subprocess.

---

##  What It Can Do

- Show a dynamic prompt with your current working directory
- Run shell commands (e.g., `ls`, `echo`, `whoami`)
- Handle built-in commands:
  - `cd` to change directory
  - `exit` or `quit` to terminate the shell session

---

##  Why It Matters

In high-performance computing (HPC), simulation clusters are controlled entirely through shell environments.  
This mini-shell introduces:
- How Slurm jobs are launched (`sbatch`)
- How command-line tools are structured
- What’s happening behind every `bash` or `terminal` interface

---

##  Clinical Research Relevance

When running:
- MRI simulation with OpenFOAM
- Surgical planning jobs with Slurm on a cluster
- Remote imaging batch processing

This project helps you understand:
- How those jobs are triggered
- How to capture logs
- How to wrap entire workflows into scriptable sessions

---

##  How to Run

```bash
cd src/
python main.py
```

You’ll see a prompt like:

```
ihunna-shell:/home/ihunna$
```

Then type:
```bash
ls
cd ..
echo Hello Shell!
exit
```

---

## Author

**Ihunna Amugo**  
DDS Candidate | MHA | MS | REHS | PhD(c) Computational Engineering  
GitHub: [@ihunnamatata](https://github.com/ihunnamatata)
