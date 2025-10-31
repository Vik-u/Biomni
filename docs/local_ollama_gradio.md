# Local Gradio Chat for Biomni with Ollama

This guide assumes you already pulled Biomni's datasets into `./biomni_full` and installed dependencies inside `.venv` (including `gradio`).

## Launch the App

```bash
source .venv/bin/activate
python gradio_app.py
```

- Opens `http://127.0.0.1:7860` in your browser.
- Responses come from Biomni's `A1` agent running against your local Ollama model (`gpt-oss:20b` by default).
- Override the host/port by exporting `GRADIO_SERVER_NAME` or `GRADIO_SERVER_PORT` (set the latter to `0` or another free port when running multiple instances).

### Optional: run in the background

```bash
source .venv/bin/activate
nohup python gradio_app.py > /tmp/biomni_gradio.log 2>&1 &
echo $!  # prints PID for later
```

- Keeps the server alive after closing the shell.
- Inspect logs with `tail -f /tmp/biomni_gradio.log`.

## Stop the App

- **Foreground run:** press `Ctrl+C` in the terminal that launched `python gradio_app.py`.
- **Background run:** terminate the PID recorded above, e.g.

  ```bash
  kill <PID>
  # or as a fallback
  pkill -f gradio_app.py
  ```

Once the process exits the UI becomes unreachable; just rerun the launch command when you're ready to start a new chat.
