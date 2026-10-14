import http from "node:http";
import { once } from "node:events";

const VERSION = "V4.40-GRANITE-SSE-CHECK";
const PROVIDER = "freeaipower";
const DEFAULT_MAX_TOKENS = 8192;
const MAX_REQUEST_BYTES = 20 * 1024 * 1024;
const MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024;
const UPSTREAM_BASE = "https://api.cloudflare.com/client/v4/accounts";

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

function sendJson(res, data, status = 200, extra = {}) {
  if (res.writableEnded || res.destroyed) return;
  res.writeHead(status, { ...corsHeaders, "Content-Type": "application/json; charset=utf-8", "X-Proxy-Version": VERSION, ...extra });
  res.end(JSON.stringify(data));
}

function sendError(res, message, status = 400, type = "invalid_request_error") {
  sendJson(res, { error: { message, type } }, status);
}

function buildModelList() {
  return Object.values(MODELS).map(model => ({ id: model.id, object: "model", created: 0, owned_by: PROVIDER }));
}

function getModelKey(name) {
  if (!name) return null;
  const value = String(name).toLowerCase();
  if (MODELS[value]) return value;
  for (const [key, model] of Object.entries(MODELS)) if (model.id.toLowerCase() === value) return key;
  if (value.includes("qwen3-30b")) return "qwen";
  if (value.includes("granite-4.0") || value.includes("ibm-granite")) return "granite";
  if (value.includes("glm-4.7-flash")) return "glm";
  if (value.includes("gemma-4-26b")) return "gemma";
  return null;
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

async function readRequestBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_REQUEST_BYTES) {
      const error = new Error("Request body exceeds 20 MB limit.");
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    const error = new Error("Invalid JSON request body.");
    error.status = 400;
    throw error;
  }
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  return messages.map(message => {
    if (!message || typeof message !== "object" || Array.isArray(message) || !Array.isArray(message.content)) return message;
    const content = message.content.map(item => {
      if (typeof item === "string") return item;
      if (!item || typeof item !== "object") return "";
      if (item.text !== undefined && item.text !== null) return String(item.text);
      if (item.content !== undefined && item.content !== null) return String(item.content);
      return "";
    }).join("\n");
    return { ...message, content };
  });
}

function inspectToolArguments(raw) {
  const value = typeof raw === "string" ? raw : String(raw ?? "");
  let parsed = null;
  let jsonValid = false;
  let nestedJsonString = false;
  try {
    parsed = JSON.parse(value);
    jsonValid = true;
    if (typeof parsed === "string") {
      try {
        const nested = JSON.parse(parsed);
        if (nested && typeof nested === "object") {
          nestedJsonString = true;
          parsed = nested;
        }
      } catch {}
    }
  } catch {}
  return {
    rawType: typeof raw,
    rawLength: value.length,
    jsonValid,
    parsedType: parsed === null ? null : Array.isArray(parsed) ? "array" : typeof parsed,
    topLevelKeys: parsed && typeof parsed === "object" && !Array.isArray(parsed) ? Object.keys(parsed).slice(0, 20) : [],
    actionsType: parsed && typeof parsed === "object" ? Array.isArray(parsed.actions) ? "array" : typeof parsed.actions : null,
    actionsCount: Array.isArray(parsed?.actions) ? parsed.actions.length : null,
    nestedJsonString
  };
}

// Normalize double-encoded JSON; repair only unambiguous trailing-comma / missing-closer cases.
function normalizeGraniteToolArguments(raw) {
  let value = typeof raw === "string" ? raw : JSON.stringify(raw ?? {});
  try {
    const parsed = JSON.parse(value);
    if (typeof parsed === "string") {
      try {
        const nested = JSON.parse(parsed);
        if (nested && typeof nested === "object") value = JSON.stringify(nested);
      } catch {}
    }
    JSON.parse(value);
    return { value, repaired: false, valid: true };
  } catch {}
  const repaired = repairTrailingJsonClosers(value);
  if (repaired !== value) {
    try {
      const parsed = JSON.parse(repaired);
      if (parsed && typeof parsed === "object") return { value: JSON.stringify(parsed), repaired: true, valid: true };
    } catch {}
  }
  return { value, repaired: false, valid: false };
}

function repairTrailingJsonClosers(input) {
  let s = String(input ?? "").trimEnd();
  if (!s) return s;
  let inString = false, escaped = false;
  const stack = [];
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') inString = true;
    else if (c === "{") stack.push("}");
    else if (c === "[") stack.push("]");
    else if (c === "}" || c === "]") {
      if (stack[stack.length - 1] === c) stack.pop();
      else return input;
    }
  }
  if (inString || stack.length === 0) return input;
  s = s.replace(/,\s*$/, "");
  for (let i = 0; i < stack.length; i++) s += stack[stack.length - 1 - i];
  return s;
}

function normalizeGraniteMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  const normalized = [];
  for (const message of messages) {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      normalized.push(message);
      continue;
    }
    const item = normalizeMessages([message])[0];
    if (item.role === "tool") {
      normalized.push({ role: "user", content: `[Tool result${item.name ? `: ${item.name}` : ""}]\n${typeof item.content === "string" ? item.content : item.content == null ? "" : String(item.content)}` });
      continue;
    }
    if (item.role === "assistant" && Array.isArray(item.tool_calls) && item.tool_calls.length) {
      const toolCalls = item.tool_calls.map(call => {
        const fn = call?.function || {};
        const normalizedArgs = normalizeGraniteToolArguments(fn.arguments);
        console.error("[GRANITE_TOOL_ARGS]", JSON.stringify({
          version: VERSION, stage: "incoming_history", toolName: fn.name || null,
          arguments: inspectToolArguments(fn.arguments), normalizedValid: normalizedArgs.valid,
          normalizedRepaired: normalizedArgs.repaired
        }));
        return { ...call, function: { ...fn, arguments: normalizedArgs.value } };
      });
      normalized.push({ ...item, content: null, tool_calls: toolCalls });
      continue;
    }
    if (typeof item.content === "string") normalized.push(item);
    else if (item.content === null || item.content === undefined) normalized.push({ ...item, content: "" });
    else normalized.push({ ...item, content: String(item.content) });
  }
  return normalized;
}

function buildUpstreamBody(body, model) {
  const key = getModelKey(model.id);
  const output = {
    ...body,
    model: model.id,
    messages: key === "granite" ? normalizeGraniteMessages(body.messages) :
      ["qwen", "glm"].includes(key) ? normalizeMessages(body.messages) : body.messages,
    stream: body.stream === true
  };
  if (model.id === "@cf/zai-org/glm-4.7-flash") delete output.reasoning_effort;
  let requested = body.max_completion_tokens ?? body.max_tokens ?? null;
  if (requested !== null) {
    requested = Number(requested);
    if (!Number.isFinite(requested) || requested <= 0) requested = null;
  }
  const resolved = Math.min(requested || DEFAULT_MAX_TOKENS, model.maxOutputTokens);
  delete output.max_tokens;
  delete output.max_completion_tokens;
  output.max_tokens = Math.floor(resolved);
  return output;
}

function normalizePrimitive(value) {
  return typeof value === "number" || typeof value === "boolean" ? String(value) : value;
}

function normalizeResponseObject(data, modelId) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return false;
  let changed = false;
  if (data.model !== modelId) {
    data.model = modelId;
    changed = true;
  }
  if (Array.isArray(data.choices)) {
    for (const choice of data.choices) {
      if (!choice || typeof choice !== "object" || Array.isArray(choice)) continue;
      for (const field of ["matched_stop", "token_ids"]) {
        if (field in choice) {
          delete choice[field];
          changed = true;
        }
      }
      for (const field of ["delta", "message"]) {
        const obj = choice[field];
        if (!obj || typeof obj !== "object" || Array.isArray(obj)) continue;
        for (const [key, value] of Object.entries(obj)) {
          if (key === "reasoning_content" && value === null) {
            delete obj[key];
            changed = true;
            continue;
          }
          if (["content", "reasoning", "reasoning_content"].includes(key) && value !== null) {
            const normalized = normalizePrimitive(value);
            if (normalized !== value) {
              obj[key] = normalized;
              changed = true;
            }
          }
        }
      }
    }
  }
  if (data.usage && typeof data.usage === "object" && "neurons" in data.usage) {
    delete data.usage.neurons;
    changed = true;
  }
  return changed;
}

function normalizeSSEFrame(frame, modelId, state) {
  const endingMatch = frame.match(/(\r?\n\r?\n)$/);
  const ending = endingMatch ? endingMatch[1] : "";
  const content = ending ? frame.slice(0, -ending.length) : frame;
  const lines = content.split(/\r?\n/);
  let changed = false;
  const normalizedLines = lines.map(line => {
    if (!line.startsWith("data:")) return line;
    const raw = line.slice(5).trim();
    if (!raw) return line;
    if (raw === "[DONE]") {
      state.receivedDone = true;
      return line;
    }
    let chunk;
    try { chunk = JSON.parse(raw); } catch { return line; }
    if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) return line;
    if (!normalizeResponseObject(chunk, modelId)) return line;
    changed = true;
    return `data: ${JSON.stringify(chunk)}`;
  });
  return changed ? normalizedLines.join("\n") + ending : frame;
}

function makeGraniteToolState() {
  return { calls: new Map(), template: null, flushed: new Set(), finishReason: null, receivedDone: false };
}

function recordGraniteToolCalls(chunk, state) {
  for (const choice of chunk?.choices || []) {
    const choiceIndex = Number(choice.index ?? 0);
    const delta = choice.delta;
    if (!delta || !Array.isArray(delta.tool_calls)) continue;
    for (const call of delta.tool_calls) {
      const toolIndex = Number(call.index ?? 0);
      const key = `${choiceIndex}:${toolIndex}`;
      let record = state.calls.get(key);
      if (!record) {
        record = { choiceIndex, toolIndex, id: "", type: "function", name: "", arguments: "", argumentChunkCount: 0 };
        state.calls.set(key, record);
      }
      if (typeof call.id === "string") record.id += call.id;
      if (typeof call.type === "string") record.type = call.type;
      if (typeof call.function?.name === "string") record.name += call.function.name;
      if (typeof call.function?.arguments === "string") {
        record.arguments += call.function.arguments;
        record.argumentChunkCount++;
      }
      state.template = chunk;
    }
  }
  for (const choice of chunk?.choices || []) {
    if (choice.finish_reason != null) state.finishReason = choice.finish_reason;
  }
}

function flushGraniteToolCalls(modelId, state) {
  const output = [];
  const grouped = new Map();
  for (const [key, call] of state.calls) {
    if (state.flushed.has(key)) continue;
    state.flushed.add(key);
    const normalized = normalizeGraniteToolArguments(call.arguments);
    console.error("[GRANITE_TOOL_ARGS]", JSON.stringify({
      version: VERSION, stage: "sse_output_check", toolName: call.name || null,
      argumentChunkCount: call.argumentChunkCount,
      accumulated: inspectToolArguments(call.arguments),
      output: inspectToolArguments(normalized.value),
      outputMatchesAccumulated: normalized.value === call.arguments,
      normalizedValid: normalized.valid,
      normalizedRepaired: normalized.repaired
    }));
    const choiceIndex = call.choiceIndex;
    if (!grouped.has(choiceIndex)) grouped.set(choiceIndex, []);
    grouped.get(choiceIndex).push({
      index: call.toolIndex,
      id: call.id || `call_${Date.now()}_${call.toolIndex}`,
      type: call.type || "function",
      function: { name: call.name, arguments: normalized.value }
    });
  }
  for (const [choiceIndex, calls] of grouped) {
    const template = state.template || {};
    output.push(`data: ${JSON.stringify({
      id: template.id,
      object: template.object || "chat.completion.chunk",
      created: template.created,
      model: modelId,
      choices: [{ index: choiceIndex, delta: { tool_calls: calls }, finish_reason: null }]
    })}\n\n`);
  }
  return output;
}

async function writeWithBackpressure(res, data) {
  if (res.destroyed || res.writableEnded) return false;
  if (res.write(data)) return true;
  await Promise.race([once(res, "drain"), once(res, "close")]).catch(() => {});
  return !res.destroyed && !res.writableEnded;
}

async function streamUpstream(upstream, res, modelId) {
  const isGranite = getModelKey(modelId) === "granite";
  res.writeHead(upstream.status, {
    ...corsHeaders,
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "X-Proxy-Version": VERSION,
    "X-Proxy-Model": modelId,
    "X-Proxy-Repaired": isGranite ? "granite-tool-arguments-checked" : "sse-normalized"
  });
  if (!upstream.body) {
    await writeWithBackpressure(res, "data: [DONE]\n\n");
    if (!res.destroyed && !res.writableEnded) res.end();
    return;
  }
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const state = { receivedDone: false };
  const graniteState = makeGraniteToolState();
  const diag = { events: 0, contentChunks: 0, reasoningChunks: 0, finishReason: null, receivedDone: false };
  let buffer = "";
  async function processFrame(frame) {
    const lines = frame.split(/\r?\n/);
    let raw = null;
    for (const line of lines) {
      if (line.startsWith("data:")) {
        raw = line.slice(5).trim();
        break;
      }
    }
    if (raw === "[DONE]") {
      if (isGranite) {
        for (const chunk of flushGraniteToolCalls(modelId, graniteState)) {
          if (!(await writeWithBackpressure(res, chunk))) return false;
        }
      }
      state.receivedDone = true;
      diag.receivedDone = true;
      if (!(await writeWithBackpressure(res, frame))) return false;
      return true;
    }
    let chunk = null;
    if (raw) {
      try { chunk = JSON.parse(raw); } catch {}
    }
    if (isGranite && chunk && typeof chunk === "object") {
      diag.events++;
      const choice = chunk.choices?.[0];
      const delta = choice?.delta;
      if (typeof delta?.content === "string" && delta.content.length) diag.contentChunks++;
      if (["reasoning_content", "reasoning"].some(k => typeof delta?.[k] === "string" && delta[k].length)) diag.reasoningChunks++;
      if (choice?.finish_reason != null) diag.finishReason = choice.finish_reason;
      recordGraniteToolCalls(chunk, graniteState);
      const hasToolDelta = (chunk.choices || []).some(c => Array.isArray(c.delta?.tool_calls) && c.delta.tool_calls.length);
      const finish = (chunk.choices || []).some(c => c.finish_reason === "tool_calls");
      if (hasToolDelta) {
        const stripped = structuredClone(chunk);
        for (const c of stripped.choices || []) {
          if (c.delta && "tool_calls" in c.delta) delete c.delta.tool_calls;
        }
        const keep = stripped.choices?.some(c => c.delta && Object.keys(c.delta).length > 0 || c.finish_reason != null);
        if (keep) {
          normalizeResponseObject(stripped, modelId);
          if (!(await writeWithBackpressure(res, `data: ${JSON.stringify(stripped)}\n\n`))) return false;
        }
        if (finish) {
          for (const out of flushGraniteToolCalls(modelId, graniteState)) {
            if (!(await writeWithBackpressure(res, out))) return false;
          }
        }
        return true;
      }
      if (finish) {
        for (const out of flushGraniteToolCalls(modelId, graniteState)) {
          if (!(await writeWithBackpressure(res, out))) return false;
        }
      }
      const normalized = normalizeSSEFrame(frame, modelId, state);
      return writeWithBackpressure(res, normalized);
    }
    return writeWithBackpressure(res, normalizeSSEFrame(frame, modelId, state));
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value || !value.length) continue;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const match = /\r\n\r\n|\n\n/.exec(buffer);
        if (!match) break;
        const end = match.index + match[0].length;
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end);
        if (!(await processFrame(frame))) {
          await reader.cancel().catch(() => {});
          return;
        }
      }
      if (Buffer.byteLength(buffer, "utf8") > MAX_SSE_EVENT_BYTES) throw new Error("SSE event exceeds the 2 MB limit.");
    }
    buffer += decoder.decode();
    if (buffer) {
      const frame = /\r?\n\r?\n$/.test(buffer) ? buffer : buffer + "\n\n";
      if (!(await processFrame(frame))) return;
    }
    if (isGranite) {
      for (const out of flushGraniteToolCalls(modelId, graniteState)) {
        if (!(await writeWithBackpressure(res, out))) return;
      }
      console.error("[GRANITE_DIAG]", JSON.stringify({
        stage: "stream_complete", version: VERSION, events: diag.events,
        contentChunks: diag.contentChunks, reasoningChunks: diag.reasoningChunks,
        finishReason: diag.finishReason, receivedDone: state.receivedDone
      }));
    }
    if (!state.receivedDone) await writeWithBackpressure(res, "data: [DONE]\n\n");
    if (!res.destroyed && !res.writableEnded) res.end();
  } catch (error) {
    console.error("SSE stream error:", error?.message || String(error));
    if (isGranite) console.error("[GRANITE_DIAG]", JSON.stringify({ stage: "stream_exception", version: VERSION, message: error?.message || String(error), ...diag }));
    if (!res.destroyed && !res.writableEnded) res.destroy(error);
    await reader.cancel().catch(() => {});
  } finally {
    reader.releaseLock();
  }
}

async function handleChatCompletions(req, res) {
  const credentials = parseCredentials(req);
  if (!credentials) return sendError(res, "Missing or invalid Authorization. Use Bearer ACCOUNT_ID:API_TOKEN.", 401);
  let body;
  try { body = await readRequestBody(req); }
  catch (error) { return sendError(res, error.message, error.status || 400); }
  if (!body || typeof body !== "object" || Array.isArray(body)) return sendError(res, "Request body must be a JSON object.", 400);
  if (!Array.isArray(body.messages)) return sendError(res, "The messages field must be an array.", 400);
  const modelKey = getModelKey(body.model);
  if (!modelKey) return sendError(res, `Unsupported model. Available models: ${Object.values(MODELS).map(m => m.id).join(", ")}`, 400);
  const model = MODELS[modelKey];
  const upstreamBody = buildUpstreamBody(body, model);
  const upstreamUrl = `${UPSTREAM_BASE}/${encodeURIComponent(credentials.accountId)}/ai/v1/chat/completions`;
  if (modelKey === "granite") {
    console.error("[GRANITE_SHAPE]", JSON.stringify({
      version: VERSION, stage: "outgoing_messages", messageCount: upstreamBody.messages.length,
      messages: upstreamBody.messages.map((m, index) => ({
        index, role: m?.role ?? null,
        contentType: m?.content === null ? "null" : Array.isArray(m?.content) ? "array" : typeof m?.content,
        hasContent: Object.prototype.hasOwnProperty.call(m || {}, "content"),
        toolCalls: (m?.tool_calls || []).map(c => ({
          functionName: c?.function?.name || null,
          arguments: inspectToolArguments(c?.function?.arguments)
        }))
      }))
    }));
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000);
  timeout.unref?.();
  const onClientClose = () => { if (!res.writableEnded) controller.abort(); };
  res.on("close", onClientClose);
  let upstream;
  try {
    upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${credentials.apiToken}`,
        "Content-Type": "application/json",
        Accept: upstreamBody.stream ? "text/event-stream" : "application/json"
      },
      body: JSON.stringify(upstreamBody),
      signal: controller.signal
    });
  } catch (error) {
    clearTimeout(timeout);
    res.off("close", onClientClose);
    if (error?.name === "AbortError" && res.destroyed) return;
    console.error("Upstream request error:", error?.message || String(error));
    return sendError(res, `Cloudflare upstream request failed: ${error?.message || String(error)}`, 502, "upstream_error");
  }
  clearTimeout(timeout);
  res.off("close", onClientClose);
  if (modelKey === "granite") {
    console.error("[GRANITE_DIAG]", JSON.stringify({
      stage: "upstream_response", version: VERSION, status: upstream.status,
      ok: upstream.ok, requestedStream: upstreamBody.stream === true,
      contentType: upstream.headers.get("content-type")
    }));
  }
  if (upstreamBody.stream === true && upstream.headers.get("content-type")?.includes("text/event-stream")) {
    return streamUpstream(upstream, res, model.id);
  }
  const raw = await upstream.text();
  let data;
  try { data = JSON.parse(raw); }
  catch {
    return sendJson(res, { error: { message: raw || `Cloudflare returned HTTP ${upstream.status}.`, type: "upstream_error" } }, upstream.ok ? 502 : upstream.status, { "X-Proxy-Model": model.id });
  }
  if (!upstream.ok) return sendJson(res, data, upstream.status, { "X-Proxy-Model": model.id });
  if (modelKey === "granite" && Array.isArray(data.choices)) {
    for (const choice of data.choices) {
      for (const call of choice?.message?.tool_calls || []) {
        const normalized = normalizeGraniteToolArguments(call?.function?.arguments);
        call.function.arguments = normalized.value;
        console.error("[GRANITE_TOOL_ARGS]", JSON.stringify({
          version: VERSION, stage: "json_output_check", toolName: call?.function?.name || null,
          output: inspectToolArguments(normalized.value), normalizedValid: normalized.valid,
          normalizedRepaired: normalized.repaired
        }));
      }
    }
  }
  normalizeResponseObject(data, model.id);
  return sendJson(res, data, upstream.status, { "X-Proxy-Model": model.id, "X-Proxy-Repaired": "json-normalized" });
}

const server = http.createServer(async (req, res) => {
  let url;
  try { url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`); }
  catch { return sendError(res, "Invalid request URL.", 400); }
  const path = url.pathname;
  if (req.method === "OPTIONS") {
    res.writeHead(204, { ...corsHeaders, "X-Proxy-Version": VERSION });
    return res.end();
  }
  if (req.method === "GET" && ["/", "/ping", "/health"].includes(path)) {
    return sendJson(res, { status: "ok", version: VERSION, provider: PROVIDER, models: Object.values(MODELS).map(model => model.id) });
  }
  if (req.method === "GET" && ["/models", "/v1/models"].includes(path)) {
    return sendJson(res, { object: "list", data: buildModelList() });
  }
  if (req.method === "POST" && ["/chat/completions", "/v1/chat/completions"].includes(path)) {
    return handleChatCompletions(req, res);
  }
  if (req.method === "GET") return sendError(res, `Not found: ${path}`, 404, "not_found_error");
  return sendError(res, `Unsupported route: ${req.method} ${path}`, 404, "not_found_error");
});

server.requestTimeout = 300000;
server.headersTimeout = 60000;
server.keepAliveTimeout = 65000;
const PORT = Number(process.env.PORT) || 3000;
server.listen(PORT, "0.0.0.0", () => {
  console.log(`${VERSION} listening on port ${PORT}; provider=${PROVIDER}; models=${Object.keys(MODELS).join(",")}`);
});
server.on("error", error => console.error("HTTP server error:", error));
