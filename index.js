import http from "node:http";
import { once } from "node:events";

const VERSION = "V4.35-OPT-RENDER";
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

function buildModelList() {
  return Object.values(MODELS).map(model => ({
    id: model.id,
    object: "model",
    created: 0,
    owned_by: PROVIDER
  }));
}

function getModelKey(name) {
  if (!name) return null;
  const value = String(name).toLowerCase();
  if (MODELS[value]) return value;
  for (const [key, model] of Object.entries(MODELS)) {
    if (model.id.toLowerCase() === value) return key;
  }
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
  const text = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(text);
  } catch {
    const error = new Error("Invalid JSON request body.");
    error.status = 400;
    throw error;
  }
}

// Normalize OpenAI text content arrays while preserving other message fields.
function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  return messages.map(message => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return message;
    const content = message.content;
    if (!Array.isArray(content)) return message;
    const normalizedContent = content.map(item => {
      if (typeof item === "string") return item;
      if (!item || typeof item !== "object") return "";
      if (item.text !== undefined && item.text !== null) return String(item.text);
      if (item.content !== undefined && item.content !== null) return String(item.content);
      return "";
    }).join("\n");
    return { ...message, content: normalizedContent };
  });
}

// Granite-specific message compatibility.
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
      const toolName = item.name ? `: ${item.name}` : "";
      const toolContent = typeof item.content === "string"
        ? item.content
        : item.content == null ? "" : String(item.content);
      normalized.push({
        role: "user",
        content: `[Tool result${toolName}]\n${toolContent}`
      });
      continue;
    }
    if (item.role === "assistant" && Array.isArray(item.tool_calls) && item.tool_calls.length > 0) {
      normalized.push({ ...item, content: null });
      continue;
    }
    if (typeof item.content === "string") {
      normalized.push(item);
    } else if (item.content === null || item.content === undefined) {
      normalized.push({ ...item, content: "" });
    } else {
      normalized.push({ ...item, content: String(item.content) });
    }
  }
  return normalized;
}

function buildUpstreamBody(body, model) {
  //console.log("Request stream:", body.stream);
  const modelKey = getModelKey(model.id);
  const output = {
    ...body,
    model: model.id,
    messages: modelKey === "granite"
      ? normalizeGraniteMessages(body.messages)
      : ["qwen", "glm"].includes(modelKey)
        ? normalizeMessages(body.messages) : body.messages,
    stream: body.stream === true
  };

  if (model.id === "@cf/zai-org/glm-4.7-flash") {
    delete output.reasoning_effort;
  }

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
  return typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : value;
}

// Returns true when any normalization changed the response.
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

      if ("matched_stop" in choice) {
        delete choice.matched_stop;
        changed = true;
      }
      if ("token_ids" in choice) {
        delete choice.token_ids;
        changed = true;
      }

      for (const field of ["delta", "message"]) {
        const obj = choice[field];
        if (!obj || typeof obj !== "object" || Array.isArray(obj)) continue;

        for (const [key, value] of Object.entries(obj)) {
          // Preserve null content and other fields; remove null reasoning
          // fields only, which may be rejected by strict clients.
          if (key === "reasoning_content" && value === null) {
            delete obj[key];
            changed = true;
            continue;
          }

          if (
            ["content", "reasoning", "reasoning_content"].includes(key) &&
            value !== null
          ) {
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
    try {
      chunk = JSON.parse(raw);
    } catch {
      return line;
    }

    if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) return line;

    if (!normalizeResponseObject(chunk, modelId)) return line;
    changed = true;
    return `data: ${JSON.stringify(chunk)}`;
  });

  return changed ? normalizedLines.join("\n") + ending : frame;
}

async function writeWithBackpressure(res, data) {
  if (res.destroyed || res.writableEnded) return false;
  if (res.write(data)) return true;

  // Stop waiting if the client disconnects before drain.
  await Promise.race([
    once(res, "drain"),
    once(res, "close")
  ]).catch(() => {});

  return !res.destroyed && !res.writableEnded;
}

async function streamUpstream(upstream, res, modelId) {
  res.writeHead(upstream.status, {
    ...corsHeaders,
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "X-Proxy-Version": VERSION,
    "X-Proxy-Model": modelId,
    "X-Proxy-Repaired": "sse-normalized"
  });

  if (!upstream.body) {
    await writeWithBackpressure(res, "data: [DONE]\n\n");
    if (!res.destroyed && !res.writableEnded) res.end();
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const state = { receivedDone: false };
  let buffer = "";

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

        const normalized = normalizeSSEFrame(frame, modelId, state);
        if (!(await writeWithBackpressure(res, normalized))) {
          await reader.cancel().catch(() => {});
          return;
        }
      }

      if (Buffer.byteLength(buffer, "utf8") > MAX_SSE_EVENT_BYTES) {
        throw new Error("SSE event exceeds the 2 MB limit.");
      }
    }

    buffer += decoder.decode();

    if (buffer) {
      // A final event may arrive without its terminating blank line.
      // Add a delimiter so the next [DONE] cannot merge into its data.
      const normalized = normalizeSSEFrame(buffer, modelId, state);
      const finalFrame = /\r?\n\r?\n$/.test(normalized)
        ? normalized
        : normalized + "\n\n";

      if (!(await writeWithBackpressure(res, finalFrame))) return;

      // A final event might itself contain [DONE].
      if (/\bdata:\s*\[DONE\]/.test(buffer)) state.receivedDone = true;
    }

    if (!state.receivedDone) {
      await writeWithBackpressure(res, "data: [DONE]\n\n");
    }

    if (!res.writableEnded && !res.destroyed) res.end();
  } catch (error) {
    console.error("SSE stream error:", error?.message || String(error));
    if (!res.destroyed && !res.writableEnded) res.destroy(error);
    await reader.cancel().catch(() => {});
  } finally {
    reader.releaseLock();
  }
}

async function handleChatCompletions(req, res) {
  const credentials = parseCredentials(req);
  if (!credentials) {
    return sendError(
      res,
      "Missing or invalid Authorization. Use Bearer ACCOUNT_ID:API_TOKEN.",
      401
    );
  }

  let body;
  try {
    body = await readRequestBody(req);
  } catch (error) {
    return sendError(res, error.message, error.status || 400);
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return sendError(res, "Request body must be a JSON object.", 400);
  }
  if (!Array.isArray(body.messages)) {
    return sendError(res, "The messages field must be an array.", 400);
  }

  const modelKey = getModelKey(body.model);
  if (!modelKey) {
    return sendError(
      res,
      `Unsupported model. Available models: ${Object.values(MODELS).map(m => m.id).join(", ")}`,
      400
    );
  }

  const model = MODELS[modelKey];
  const upstreamBody = buildUpstreamBody(body, model);
  const upstreamUrl =
    `${UPSTREAM_BASE}/${encodeURIComponent(credentials.accountId)}/ai/v1/chat/completions`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000);
  timeout.unref?.();

  // Abort the upstream request when the client disconnects.
  const onClientClose = () => {
    if (!res.writableEnded) controller.abort();
  };
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
    return sendError(
      res,
      `Cloudflare upstream request failed: ${error?.message || String(error)}`,
      502,
      "upstream_error"
    );
  }

  clearTimeout(timeout);
  res.off("close", onClientClose);

  const isStreaming = upstreamBody.stream === true;
  if (
    isStreaming &&
    upstream.headers.get("content-type")?.includes("text/event-stream")
  ) {
    return streamUpstream(upstream, res, model.id);
  }

  const raw = await upstream.text();
  let data;

  try {
    data = JSON.parse(raw);
  } catch {
    return sendJson(
      res,
      {
        error: {
          message: raw || `Cloudflare returned HTTP ${upstream.status}.`,
          type: "upstream_error"
        }
      },
      upstream.ok ? 502 : upstream.status,
      { "X-Proxy-Model": model.id }
    );
  }

  if (!upstream.ok) {
    // Keep Cloudflare's original error details for diagnosis.
    return sendJson(res, data, upstream.status, { "X-Proxy-Model": model.id });
  }

  normalizeResponseObject(data, model.id);
  return sendJson(res, data, upstream.status, {
    "X-Proxy-Model": model.id,
    "X-Proxy-Repaired": "json-normalized"
  });
}

const server = http.createServer(async (req, res) => {
  let url;

  try {
    url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  } catch {
    return sendError(res, "Invalid request URL.", 400);
  }

  const path = url.pathname;

  if (req.method === "OPTIONS") {
    res.writeHead(204, { ...corsHeaders, "X-Proxy-Version": VERSION });
    return res.end();
  }

  if (req.method === "GET" && ["/", "/ping", "/health"].includes(path)) {
    return sendJson(res, {
      status: "ok",
      version: VERSION,
      provider: PROVIDER,
      models: Object.values(MODELS).map(model => model.id)
    });
  }

  if (req.method === "GET" && ["/models", "/v1/models"].includes(path)) {
    return sendJson(res, { object: "list", data: buildModelList() });
  }

  if (
    req.method === "POST" &&
    ["/chat/completions", "/v1/chat/completions"].includes(path)
  ) {
    return handleChatCompletions(req, res);
  }

  if (req.method === "GET") {
    return sendError(res, `Not found: ${path}`, 404, "not_found_error");
  }

  return sendError(
    res,
    `Unsupported route: ${req.method} ${path}`,
    404,
    "not_found_error"
  );
});

server.requestTimeout = 300000;
server.headersTimeout = 60000;
server.keepAliveTimeout = 65000;

const PORT = Number(process.env.PORT) || 3000;

server.listen(PORT, "0.0.0.0", () => {
  console.log(
    `${VERSION} listening on port ${PORT}; provider=${PROVIDER}; models=${Object.keys(MODELS).join(",")}`
  );
});

server.on("error", error => {
  console.error("HTTP server error:", error);
});
