import http from "node:http";
import { randomUUID } from "node:crypto";

const VERSION = "V4.35-OPT-RENDER";
const PROVIDER = "freeaipower";
const DEFAULT_MAX_TOKENS = 8192;
const MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024;
const MAX_REQUEST_BYTES = 20 * 1024 * 1024;

const MODELS = {
  qwen: { id: "@cf/qwen/qwen3-30b-a3b-fp8", contextWindow: 32768, maxOutputTokens: 8192 },
  granite: { id: "@cf/ibm-granite/granite-4.0-h-micro", contextWindow: 131000, maxOutputTokens: 8192 },
  glm: { id: "@cf/zai-org/glm-4.7-flash", contextWindow: 131072, maxOutputTokens: 8192 },
  gemma: { id: "@cf/google/gemma-4-26b-a4b-it", contextWindow: 262144, maxOutputTokens: 8192 }
};

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Authorization,Content-Type,Accept,Origin,X-Requested-With",
  "Access-Control-Expose-Headers": "Content-Type,X-Proxy-Version,X-Proxy-Model,X-Proxy-Repaired"
};

const CONTENT_PRIMITIVE_RE = /"content"\s*:\s*(?:-?\d|true\b|false\b)/;

function sendJson(res, data, status = 200, extra = {}) {
  if (res.destroyed) return;
  res.writeHead(status, {
    ...corsHeaders,
    "Content-Type": "application/json; charset=utf-8",
    "X-Proxy-Version": VERSION,
    ...extra
  });
  res.end(JSON.stringify(data));
}

function sendError(res, message, status = 400, type = "invalid_request_error") {
  sendJson(res, { error: { message, type } }, status);
}

function parseCredentials(req) {
  const auth = req.headers.authorization || "";
  if (!auth.toLowerCase().startsWith("bearer ")) return null;

  const token = auth.slice(7).trim();
  const separator = token.indexOf(":");
  if (separator <= 0 || separator === token.length - 1) return null;

  const accountId = token.slice(0, separator).trim();
  const apiToken = token.slice(separator + 1).trim();

  return accountId && apiToken ? { accountId, apiToken } : null;
}

function getModelKey(name) {
  if (!name) return null;
  const value = String(name).toLowerCase();

  if (MODELS[value]) return value;

  for (const [key, model] of Object.entries(MODELS)) {
    if (model.id.toLowerCase() === value) return key;
  }

  if (value.includes("qwen3-30b")) return "qwen";
  if (value.includes("granite-4.0")) return "granite";
  if (value.includes("glm-4.7-flash")) return "glm";
  if (value.includes("gemma-4-26b")) return "gemma";

  return null;
}

/*
 * Cherry Studio compatibility normalization:
 * - Canonicalize model IDs.
 * - Convert primitive delta.content to strings.
 * - Remove null-valued delta fields, which can violate strict schemas.
 * - Remove nonstandard token_ids and neurons fields.
 * - Preserve non-null reasoning fields and normal content.
 */
function normalizeChatResponseObject(chunk, modelId) {
  if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) {
    return { chunk, changed: false };
  }

  let changed = false;

  if (modelId && chunk.model !== modelId) {
    chunk.model = modelId;
    changed = true;
  }

  if (chunk.usage && typeof chunk.usage === "object" && !Array.isArray(chunk.usage)) {
    if (Object.hasOwn(chunk.usage, "neurons")) {
      delete chunk.usage.neurons;
      changed = true;
    }
  }

  if (!Array.isArray(chunk.choices)) return { chunk, changed };

  for (const choice of chunk.choices) {
    if (!choice || typeof choice !== "object") continue;

    if (Object.hasOwn(choice, "token_ids")) {
      delete choice.token_ids;
      changed = true;
    }

    const delta = choice.delta;
    if (!delta || typeof delta !== "object" || Array.isArray(delta)) continue;

    for (const key of Object.keys(delta)) {
      if (delta[key] === null) {
        delete delta[key];
        changed = true;
      }
    }

    if (typeof delta.content === "number" || typeof delta.content === "boolean") {
      delta.content = String(delta.content);
      changed = true;
    }
  }

  return { chunk, changed };
}

function normalizeUsageOnlyChunk(chunk, modelId) {
  if (!chunk || typeof chunk !== "object") return chunk;

  if (
    chunk.choices === undefined &&
    chunk.usage &&
    typeof chunk.usage === "object" &&
    !Array.isArray(chunk.usage)
  ) {
    return {
      id: chunk.id || `chatcmpl-${randomUUID()}`,
      object: "chat.completion.chunk",
      created: chunk.created || Math.floor(Date.now() / 1000),
      model: modelId,
      choices: [],
      usage: chunk.usage
    };
  }

  return chunk;
}

function normalizeResponse(chunk, modelId) {
  const normalized = normalizeChatResponseObject(chunk, modelId).chunk;
  return normalizeUsageOnlyChunk(normalized, modelId);
}

function estimateTokens(messages) {
  let chars = 0;

  for (const message of messages || []) {
    if (!message) continue;

    const content = message.content;

    if (typeof content === "string") {
      chars += content.length;
    } else if (Array.isArray(content)) {
      for (const item of content) {
        if (!item) continue;
        if (typeof item.text === "string") chars += item.text.length;
        if (typeof item.input_text === "string") chars += item.input_text.length;
      }
    } else if (content != null) {
      chars += String(content).length;
    }

    if (typeof message.role === "string") chars += message.role.length;
    if (typeof message.name === "string") chars += message.name.length;
  }

  return Math.ceil(chars / 4);
}

function buildUpstreamBody(body, model) {
  const output = { ...body, model: model.id, stream: body.stream !== false };

  let requested = body.max_completion_tokens ?? body.max_tokens ?? null;

  if (requested !== null && requested !== undefined) {
    requested = Number(requested);
    if (!Number.isFinite(requested) || requested <= 0) requested = null;
  }

  const resolved = Math.min(requested || DEFAULT_MAX_TOKENS, model.maxOutputTokens);

  delete output.max_tokens;
  delete output.max_completion_tokens;
  output.max_tokens = resolved;

  return {
    upstreamBody: output,
    requestedMaxTokens: requested,
    resolvedMaxTokens: resolved
  };
}

function buildModelList() {
  return Object.values(MODELS).map(model => ({
    id: model.id,
    object: "model",
    created: 0,
    owned_by: "cloudflare"
  }));
}

function findSSEDelimiter(text) {
  const lf = text.indexOf("\n\n");
  const crlf = text.indexOf("\r\n\r\n");

  if (lf < 0 && crlf < 0) return null;
  if (crlf >= 0 && (lf < 0 || crlf <= lf)) return { index: crlf, length: 4 };

  return { index: lf, length: 2 };
}

function processSSEFrame(frame, modelId, stats) {
  const lines = frame.split(/\r?\n/);
  let dataIndex = -1;
  let dataText = "";

  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith("data:")) continue;
    dataIndex = i;
    dataText = lines[i].slice(5).trim();
    break;
  }

  if (dataIndex < 0 || !dataText) return frame;

  if (dataText === "[DONE]") {
    stats.receivedDone = true;
    return frame;
  }

  const needsContentFix = CONTENT_PRIMITIVE_RE.test(dataText);
  let chunk;

  try {
    chunk = JSON.parse(dataText);
  } catch {
    return frame;
  }

  if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) return frame;

  const normalized = normalizeResponse(chunk, modelId);
  const changed = needsContentFix || JSON.stringify(normalized) !== JSON.stringify(chunk);

  if (!changed) return frame;

  stats.normalizedChunks++;

  if (needsContentFix) stats.fixedContent++;
  return `data: ${JSON.stringify(normalized)}\n\n`;
}

function writeWithBackpressure(res, data) {
  if (res.destroyed || res.writableEnded) return Promise.resolve(false);
  if (res.write(data)) return Promise.resolve(true);

  return new Promise(resolve => {
    const done = () => {
      res.off("drain", onDrain);
      res.off("close", onClose);
      resolve(!res.destroyed);
    };

    const onDrain = () => done();
    const onClose = () => done();

    res.once("drain", onDrain);
    res.once("close", onClose);
  });
}

async function streamSSE(upstreamBody, res, modelKey, modelId) {
  const reader = upstreamBody.getReader();
  const decoder = new TextDecoder();

  const stats = {
    events: 0,
    inputBytes: 0,
    outputBytes: 0,
    rawFlushes: 0,
    fixedContent: 0,
    normalizedChunks: 0,
    receivedDone: false
  };

  let pending = "";
  let bypassUntilDelimiter = false;
  let doneProbe = "";
  let finished = false;

  async function writeText(text) {
    if (!text) return !res.destroyed;
    stats.outputBytes += Buffer.byteLength(text);
    return writeWithBackpressure(res, text);
  }

  async function handleText(text) {
    if (!text) return;

    doneProbe = (doneProbe + text).slice(-32);
    if (doneProbe.includes("[DONE]")) stats.receivedDone = true;

    let buffer = pending + text;
    pending = "";

    while (buffer.length && !res.destroyed) {
      const delimiter = findSSEDelimiter(buffer);

      if (bypassUntilDelimiter) {
        if (delimiter) {
          const end = delimiter.index + delimiter.length;
          if (!await writeText(buffer.slice(0, end))) return;
          stats.events++;
          buffer = buffer.slice(end);
          bypassUntilDelimiter = false;
          continue;
        }

        if (buffer.length > 3) {
          if (!await writeText(buffer.slice(0, -3))) return;
          buffer = buffer.slice(-3);
        }

        pending = buffer;
        return;
      }

      if (delimiter) {
        const end = delimiter.index + delimiter.length;
        const frame = buffer.slice(0, end);
        stats.events++;

        if (!await writeText(processSSEFrame(frame, modelId, stats))) return;

        buffer = buffer.slice(end);
        continue;
      }

      if (buffer.length > MAX_SSE_EVENT_BYTES) {
        const safeLength = buffer.length - 3;
        if (!await writeText(buffer.slice(0, safeLength))) return;

        pending = buffer.slice(safeLength);
        bypassUntilDelimiter = true;
        stats.rawFlushes++;
        return;
      }

      pending = buffer;
      return;
    }
  }

  try {
    while (!res.destroyed) {
      const result = await reader.read();
      if (result.done) break;

      const value = result.value;
      if (!value || !value.length) continue;

      stats.inputBytes += value.length;
      await handleText(decoder.decode(value, { stream: true }));
    }

    if (!res.destroyed) {
      const finalText = decoder.decode();
      if (finalText) await handleText(finalText);

      if (pending) {
        if (bypassUntilDelimiter) {
          await writeText(pending);
        } else {
          stats.events++;
          await writeText(processSSEFrame(pending, modelId, stats));
        }
        pending = "";
      }

      if (!stats.receivedDone) await writeText("data: [DONE]\n\n");

      finished = true;

      console.log("V4.35 STREAM END", JSON.stringify({
        model: modelKey,
        events: stats.events,
        inputBytes: stats.inputBytes,
        outputBytes: stats.outputBytes,
        rawFlushes: stats.rawFlushes,
        fixedContent: stats.fixedContent,
        normalizedChunks: stats.normalizedChunks
      }));

      if (!res.destroyed && !res.writableEnded) res.end();
    }
  } catch (error) {
    console.error("V4.35 STREAM ERROR", JSON.stringify({
      model: modelKey,
      message: error?.message || String(error),
      inputBytes: stats.inputBytes,
      events: stats.events
    }));

    if (!res.destroyed && !res.writableEnded) {
      if (!res.headersSent) {
        sendError(res, "Cloudflare stream processing failed.", 502, "upstream_error");
      } else {
        res.destroy(error);
      }
    }
  } finally {
    if (!finished || res.destroyed) {
      try {
        await reader.cancel();
      } catch {}
    }

    try {
      reader.releaseLock();
    } catch {}
  }
}

async function readRequestJson(req) {
  const chunks = [];
  let size = 0;

  for await (const chunk of req) {
    size += chunk.length;

    if (size > MAX_REQUEST_BYTES) {
      throw Object.assign(new Error("Request body exceeds 20 MB limit."), { status: 413 });
    }

    chunks.push(chunk);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw Object.assign(new Error("Invalid JSON body."), { status: 400 });
  }
}

async function handleChatCompletions(req, res) {
  const credentials = parseCredentials(req);

  if (!credentials) {
    return sendError(res, "Missing or invalid Authorization. Use Bearer ACCOUNT_ID:API_TOKEN.", 401);
  }

  let body;

  try {
    body = await readRequestJson(req);
  } catch (error) {
    return sendError(res, error.message, error.status || 400);
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return sendError(res, "Request body must be a JSON object.", 400);
  }

  const modelKey = getModelKey(body.model);
  const model = modelKey ? MODELS[modelKey] : null;

  if (!model) {
    return sendError(res, `Unsupported model: ${body.model || "(missing)"}`, 400);
  }

  if (!Array.isArray(body.messages)) {
    return sendError(res, "messages must be an array.", 400);
  }

  const inputTokens = estimateTokens(body.messages);

  if (inputTokens >= model.contextWindow) {
    return sendError(
      res,
      `Input exceeds context window. Estimated input tokens: ${inputTokens}, context window: ${model.contextWindow}.`,
      400
    );
  }

  const { upstreamBody, requestedMaxTokens, resolvedMaxTokens } = buildUpstreamBody(body, model);

  console.log("V4.35 REQUEST", JSON.stringify({
    model: model.id,
    modelKey,
    stream: upstreamBody.stream,
    inputTokens,
    messageCount: body.messages.length,
    requestedMaxTokens,
    resolvedMaxTokens
  }));

  const upstreamUrl =
    `https://api.cloudflare.com/client/v4/accounts/` +
    `${encodeURIComponent(credentials.accountId)}/ai/v1/chat/completions`;

  let upstreamResponse;

  try {
    upstreamResponse = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${credentials.apiToken}`,
        "Content-Type": "application/json",
        "Accept": upstreamBody.stream ? "text/event-stream" : "application/json"
      },
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(600000)
    });
  } catch (error) {
    console.error("V4.35 UPSTREAM FETCH ERROR", JSON.stringify({
      model: model.id,
      message: error?.message || String(error)
    }));

    return sendError(res, "Cloudflare upstream connection failed or timed out.", 502, "upstream_error");
  }

  if (!upstreamResponse.ok) {
    const errorText = await upstreamResponse.text();

    console.error("V4.35 UPSTREAM ERROR", JSON.stringify({
      model: model.id,
      status: upstreamResponse.status,
      body: errorText.slice(0, 1000)
    }));

    res.writeHead(upstreamResponse.status, {
      ...corsHeaders,
      "Content-Type": upstreamResponse.headers.get("Content-Type") || "application/json; charset=utf-8",
      "X-Proxy-Version": VERSION,
      "X-Proxy-Model": model.id,
      "X-Proxy-Repaired": "0"
    });

    return res.end(errorText || JSON.stringify({
      error: { message: `Cloudflare Workers AI returned HTTP ${upstreamResponse.status}.` }
    }));
  }

  if (upstreamBody.stream === false) {
    let result;

    try {
      result = await upstreamResponse.json();
    } catch {
      return sendError(res, "Cloudflare Workers AI returned invalid JSON.", 502, "upstream_error");
    }

    result = normalizeResponse(result, model.id);

    return sendJson(res, result, 200, {
      "X-Proxy-Model": model.id,
      "X-Proxy-Repaired": "1"
    });
  }

  if (!upstreamResponse.body) {
    return sendError(res, "Cloudflare Workers AI returned an empty stream.", 502, "upstream_error");
  }

  res.writeHead(upstreamResponse.status, {
    ...corsHeaders,
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "X-Proxy-Version": VERSION,
    "X-Proxy-Model": model.id,
    "X-Proxy-Repaired": "stream"
  });

  res.flushHeaders?.();
  await streamSSE(upstreamResponse.body, res, modelKey, model.id);
}

async function handleRequest(req, res) {
  if (req.method === "OPTIONS") {
    res.writeHead(204, corsHeaders);
    return res.end();
  }

  const path = new URL(req.url, "http://localhost").pathname;

  if (req.method === "GET" && (path === "/" || path === "/ping")) {
    return sendJson(res, {
      status: "ok",
      version: VERSION,
      service: "freeaipower OpenAI Compatible Proxy",
      owned_by: PROVIDER,
      models: Object.values(MODELS).map(model => model.id)
    });
  }

  if (req.method === "GET" && (path === "/models" || path === "/v1/models")) {
    return sendJson(res, {
      object: "list",
      data: buildModelList()
    });
  }

  if (req.method === "POST" && (path === "/chat/completions" || path === "/v1/chat/completions")) {
    return handleChatCompletions(req, res);
  }

  return sendError(res, "Not Found", 404);
}

const server = http.createServer((req, res) => {
  handleRequest(req, res).catch(error => {
    console.error("V4.35 INTERNAL ERROR", error?.stack || error?.message || String(error));

    if (res.destroyed || res.writableEnded) return;

    if (!res.headersSent) {
      sendError(res, "Internal Server Error", 500, "internal_error");
    } else {
      res.destroy(error);
    }
  });
});

server.requestTimeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

const PORT = Number(process.env.PORT) || 3000;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`${VERSION} listening on 0.0.0.0:${PORT}`);
});

