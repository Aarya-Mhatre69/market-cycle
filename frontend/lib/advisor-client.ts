import type { AdvisorErrorKind, AskAdvisorResponse } from "./types";

/**
 * Base URL of the FastAPI server (main.py). Configure via
 * NEXT_PUBLIC_SHANKH_API_URL in your Next.js environment — falls back to
 * localhost for local dev so the app never silently points nowhere.
 */
const apiUrl = process.env.NEXT_PUBLIC_SHANKH_API_URL;

if (!apiUrl) {
  throw new Error(
    "Missing NEXT_PUBLIC_SHANKH_API_URL environment variable."
  );
}

export const ADVISOR_API_URL = apiUrl.replace(/\/+$/, "");

/** The deep agent fans out to three subagents — give it real headroom. */
const DEFAULT_ASK_TIMEOUT_MS = 120_000;
const HEALTH_TIMEOUT_MS = 6_000;

export class AdvisorApiError extends Error {
  kind: AdvisorErrorKind;
  status?: number;

  constructor(message: string, kind: AdvisorErrorKind, status?: number) {
    super(message);
    this.name = "AdvisorApiError";
    this.kind = kind;
    this.status = status;
  }
}

function isAbortError(err: unknown): err is DOMException {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Merge a caller-provided AbortSignal with an internal timeout signal. */
function withTimeout(externalSignal: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let timedOut = false;

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    else externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  return {
    signal: controller.signal,
    cleanup: () => clearTimeout(timer),
    wasTimeout: () => timedOut,
    wasExternallyAborted: () => externalSignal?.aborted ?? false,
  };
}

async function safeParseJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return undefined; // signals "was not valid JSON"
  }
}

export interface AskAdvisorOptions {
  question: string;
  threadId: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * Calls POST /ask. Every failure mode the FastAPI server can produce
 * (offline, 503 not-yet-initialized, 500 agent error, slow response,
 * malformed body) is normalized into an AdvisorApiError so the UI only
 * ever has one thing to branch on.
 */
export async function askAdvisor({
  question,
  threadId,
  signal,
  timeoutMs = DEFAULT_ASK_TIMEOUT_MS,
}: AskAdvisorOptions): Promise<string> {
  const trimmed = question.trim();
  if (!trimmed) {
    throw new AdvisorApiError("Question cannot be empty.", "invalid_response");
  }

  const { signal: combinedSignal, cleanup, wasTimeout, wasExternallyAborted } = withTimeout(
    signal,
    timeoutMs
  );

  let res: Response;
  try {
    res = await fetch(`${ADVISOR_API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed, thread_id: threadId }),
      signal: combinedSignal,
    });
  } catch (err) {
    cleanup();
    if (isAbortError(err)) {
      if (wasExternallyAborted()) {
        throw new AdvisorApiError("Request cancelled.", "aborted");
      }
      if (wasTimeout()) {
        throw new AdvisorApiError(
          "Shankh is taking longer than expected to respond. It may still finish server-side.",
          "timeout"
        );
      }
      throw new AdvisorApiError("Request cancelled.", "aborted");
    }
    throw new AdvisorApiError(
      `Could not reach the advisor service at ${ADVISOR_API_URL}. Confirm the FastAPI server is running and reachable.`,
      "network"
    );
  }
  cleanup();

  if (res.status === 503) {
    throw new AdvisorApiError(
      "Shankh is still starting up on the server. Try again in a moment.",
      "unavailable",
      503
    );
  }

  if (!res.ok) {
    const body = await safeParseJson(res);
    const detail =
      body && typeof body === "object" && "detail" in body && typeof (body as any).detail === "string"
        ? (body as any).detail
        : `Advisor request failed (HTTP ${res.status}).`;
    throw new AdvisorApiError(detail, "server", res.status);
  }

  const body = await safeParseJson(res);
  if (body === undefined) {
    throw new AdvisorApiError("The server returned a response that wasn't valid JSON.", "invalid_response");
  }
  if (!body || typeof (body as Partial<AskAdvisorResponse>).response !== "string") {
    throw new AdvisorApiError("The server response was missing the expected 'response' field.", "invalid_response");
  }

  return (body as AskAdvisorResponse).response;
}

/** GET /health — used purely for the connection badge, never blocks sending. */
export async function checkAdvisorHealth(signal?: AbortSignal): Promise<boolean> {
  const { signal: combinedSignal, cleanup } = withTimeout(signal, HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(`${ADVISOR_API_URL}/health`, {
      signal: combinedSignal,
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    cleanup();
  }
}