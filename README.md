# local-dictation

Push-to-talk dictation for Windows that runs entirely on your own machine. Press a key or a mouse side button,
speak (Chinese, English, or both mixed in one sentence), and get clean written text where you want it.

- **Speech to text:** Whisper large-v3 (faster-whisper on an NVIDIA GPU)
- **Clean-up and rewriting:** Qwen (`qwen3.5:9b`) through a local Ollama server
- **Nothing leaves the computer:** no cloud calls, no audio upload

## What it does

| Trigger | Mode | Result |
|---|---|---|
| **F10** or **mouse front side button** | Dictation | Your speech as written text. Refinement level Light / Medium / Strong (Settings). |
| **Ctrl + F10** or **Ctrl + side button** | Prompt | Your speech organized into a clear prompt for an AI assistant. |
| **Shift + side button** | Assistant | A spoken instruction carried out on the text you selected (revise, shorten, translate, reply, explain...). |

Press the same key again to stop; **Esc** cancels a recording.

**The bubble.** A small status pill (bottom-right; drag it anywhere) grows upward into a panel. It shows a rough
live transcript while you speak, then what was heard and the result.

- **Dictation and prompt mode:** fix the result in the bubble if needed, then **middle-click** where it should go,
  press **F10** to paste at the cursor, or click **Paste**. **✕** discards it.
- **Assistant mode:** the result stays in the bubble with Paste / Copy / Retry. Press Shift + side button again to
  change it by voice. **Ctrl+Z** in the bubble goes back to the previous version.

**Personal vocabulary.** When you fix a misheard term in the bubble (e.g. "Hitmap" → "Heatmap"), it is added to
`vocabulary.txt` with how it was misheard. The model reads that file as separate notes and uses them where the
context fits. Nothing is replaced automatically, and you can edit the file at any time (tray → Vocabulary...).
Content edits (numbers, "this" → "that", rewrites) are not learned. For Chinese, a change is only learned if both
versions sound alike in pinyin.

**Tray icon.** Settings (refinement level, bubble opacity, learning, live preview), History, Vocabulary, Quit.
Double-clicking the desktop shortcut while dictation is running offers to stop it.

## Requirements

- Windows 10 or 11
- An NVIDIA GPU with about 12 GB of free VRAM (Whisper large-v3 fp16 plus Qwen 9B), with a recent driver
- Python 3.11
- A microphone (a headset works best)

## Setup

The model files and Ollama live in one folder. It is `D:\LLM` by default; to use another folder, set the
environment variable `DICTATE_LLM_HOME`. Put this repository anywhere, for example `D:\LLM\dictate`.

1. **Ollama.** Download `ollama-windows-amd64.zip` from https://github.com/ollama/ollama/releases, extract it to
   `D:\LLM\ollama`, then pull the model:
   ```bat
   set OLLAMA_MODELS=D:\LLM\models
   D:\LLM\ollama\ollama.exe serve
   :: in a second window:
   D:\LLM\ollama\ollama.exe pull qwen3.5:9b
   ```
   You do not need to start Ollama yourself later: the app starts it when needed.

2. **Python environment.**
   ```bat
   cd D:\LLM\dictate
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

3. **Whisper model** into `D:\LLM\whisper\faster-whisper-large-v3`.
   - **If huggingface.co is reachable:**
     ```bat
     .venv\Scripts\python -c "from faster_whisper.utils import download_model; download_model('large-v3', output_dir=r'D:\LLM\whisper\faster-whisper-large-v3')"
     ```
   - **If Hugging Face is blocked:** download OpenAI's official weights and convert them offline with
     `tools/convert_whisper.py` (instructions are in that file). The weights are at
     https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt;
     the long hex string in the URL is the file's SHA-256. Save the file as `D:\LLM\whisper\large-v3.pt`.

4. **Run it.**
   ```bat
   .venv\Scripts\pythonw.exe dictate.py
   ```
   It takes about 30 seconds to load; the bubble says "Dictation ready".

   To start it at login, put a shortcut in `shell:startup` with:
   - Target: `D:\LLM\dictate\.venv\Scripts\pythonw.exe "D:\LLM\dictate\dictate.py"`
   - Start in: `D:\LLM\dictate`
   - Icon: `dictate.ico`

## Files it writes (local only, never committed)

| File | Contents |
|---|---|
| `history.jsonl` | Every dictation: transcript, result, your edits |
| `recent/` | The last 10 recordings, for troubleshooting |
| `failures/` | Recordings where recognition misfired |
| `vocabulary.txt` | Your terms and how they were misheard |
| `settings.json` | Your settings |
| `dictate.log` | Log |

`history.jsonl`, `recent/`, `failures/` and `vocabulary.txt` contain what you dictated. Treat them like any other
sensitive notes; `.gitignore` keeps them out of the repository.

## Notes

- **F12 cannot be used as the hotkey:** Windows reserves it for the debugger.
- **Languages:** only Chinese and English are recognized; on very little speech Whisper otherwise guesses other
  languages.
- **Quiet speech:** the voice-activity threshold is lowered (0.25) for quiet voices. If long dictations still come
  out short, raise the Windows microphone input level.
- **Assistant mode:** with nothing selected, the 9B model may invent details. Review the result before pasting.
- **Console windows:** selected text in a console (e.g. Windows Terminal) is not read in assistant mode, because
  Ctrl+C there would interrupt the running program.

## License

MIT (see `LICENSE`). `tools/convert_openai_to_hf.py` is unmodified from Hugging Face transformers v4.46.0 and stays
under its Apache License 2.0 header. The models (Whisper, Qwen) and Ollama have their own licenses.
