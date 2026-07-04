from __future__ import annotations

import os
import sys

# Same bitsandbytes-shadow guard as training: running from the repo root shadows the real bnb.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd(), _HERE)]

import copy
import json
import threading

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

from Reasearch_Drug_Chatbot import (
    GemmaRagEngine,
    ServingConfig,
    configure_logging,
    format_rag_messages,
)

API_TOKEN = os.environ.get("RAG_API_TOKEN", "")  # if set, callers must send Authorization: Bearer <token>
CORS_ORIGINS = [o.strip() for o in os.environ.get("RAG_CORS_ORIGINS", "*").split(",") if o.strip()]

configure_logging()
engine = GemmaRagEngine(ServingConfig()).load()  # loads base + LoRA adapter + retriever once, at startup
GEN_LOCK = threading.Lock()  # the 12B can only do one generate at a time; serialize requests

app = FastAPI(title="Drug Information RAG Chatbot")
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["POST", "GET"], allow_headers=["*"],
)


class ChatIn(BaseModel):
    question: str
    max_new_tokens: int | None = None


def require_token(request: Request) -> None:
    if API_TOKEN and request.headers.get("authorization", "") != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing API token")


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def start_stream(question: str, max_new_tokens: int | None):
    # Mirror engine.generate but attach a streamer so tokens come out live.
    context_blocks = engine.retrieve_context(question)
    messages = format_rag_messages(question=question, context_blocks=context_blocks)
    text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = engine.processor(text=text, return_tensors="pt", add_special_tokens=False).to(engine.device)
    gen_config = engine.generation_config
    if max_new_tokens:
        gen_config = copy.deepcopy(gen_config)
        gen_config.max_new_tokens = max_new_tokens
    tok = getattr(engine.processor, "tokenizer", None) or engine.processor
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def run():
        with torch.inference_mode():
            engine.model.generate(**inputs, generation_config=gen_config, streamer=streamer)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return streamer, thread


@app.get("/health")
def health():
    return {"ok": True, "adapter": engine.adapter_dir, "retrieval": bool(engine.retriever)}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    require_token(request)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    with GEN_LOCK:
        answer = engine.answer(question, max_new_tokens=body.max_new_tokens)
    return {"answer": answer}


@app.post("/chat/stream")
def chat_stream(body: ChatIn, request: Request):
    require_token(request)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    def event_stream():
        with GEN_LOCK:
            streamer, thread = start_stream(question, body.max_new_tokens)
            try:
                for token in streamer:
                    yield sse({"token": token})
            finally:
                thread.join()
            yield sse({"done": True})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drug Information RAG Chatbot</title>
<style>
  :root { --bg:#0f1419; --panel:#1a212b; --user:#2b6cb0; --bot:#232c38; --text:#e6edf3; --muted:#8b98a5; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 18px; background:var(--panel); font-weight:600; border-bottom:1px solid #2a3542; }
  header small { display:block; font-weight:400; color:var(--muted); font-size:12px; margin-top:2px; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:min(720px, 92%); padding:11px 14px; border-radius:12px; line-height:1.5; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--user); }
  .bot  { align-self:flex-start; background:var(--bot); border:1px solid #2a3542; }
  form { display:flex; gap:8px; padding:12px; background:var(--panel); border-top:1px solid #2a3542; }
  input { flex:1; padding:12px 14px; border-radius:10px; border:1px solid #2a3542; background:#0f1419; color:var(--text); font-size:15px; }
  button { padding:0 20px; border:0; border-radius:10px; background:var(--user); color:#fff; font-size:15px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
</style>
</head>
<body>
  <header>Drug Information RAG Chatbot
    <small>Answers are grounded in retrieved literature and cite PMIDs. Educational information only, not medical advice.</small>
  </header>
  <div id="log"></div>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="Ask about a drug, mechanism, trial, or finding..." />
    <button id="send" type="submit">Send</button>
  </form>
<script>
  const log = document.getElementById('log'), form = document.getElementById('f'),
        q = document.getElementById('q'), send = document.getElementById('send');
  function bubble(cls, text){ const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text; log.appendChild(d); log.scrollTop=log.scrollHeight; return d; }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = q.value.trim(); if(!question) return;
    bubble('user', question); q.value=''; send.disabled=true;
    const out = bubble('bot', ''); let got='';
    try {
      const res = await fetch('chat/stream', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question}) });
      if(!res.ok || !res.body) throw new Error('stream failed');
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf='';
      while(true){
        const {value, done} = await reader.read(); if(done) break;
        buf += dec.decode(value, {stream:true});
        let i; while((i = buf.indexOf('\\n\\n')) >= 0){
          const line = buf.slice(0, i).trim(); buf = buf.slice(i+2);
          if(line.startsWith('data:')){ const p = JSON.parse(line.slice(5).trim());
            if(p.token){ got += p.token; out.textContent = got; log.scrollTop = log.scrollHeight; } }
        }
      }
    } catch(err){
      try { const r = await fetch('chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question}) });
            const j = await r.json(); out.textContent = j.answer || ('Error: '+(j.detail||'request failed')); }
      catch(e2){ out.textContent = 'Error: could not reach the server.'; }
    } finally { send.disabled=false; q.focus(); }
  });
</script>
</body>
</html>"""


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("RAG_HOST", "0.0.0.0"), port=int(os.environ.get("RAG_PORT", "8000")))


if __name__ == "__main__":
    main()
