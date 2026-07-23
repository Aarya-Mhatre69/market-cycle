"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AdvisorApiError, askAdvisor, checkAdvisorHealth } from "@/lib/advisor-client";
import type { ChatMessage, ConnectionState } from "@/lib/types";

const STORAGE_PREFIX = "shankh-advisor:";
const HEALTH_POLL_MS = 20_000;
const HEALTH_RETRY_MS = 6_000; // faster retry cadence while known offline

function newThreadId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newMessageId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadThreadId(): string {
  try {
    const stored = window.localStorage.getItem(`${STORAGE_PREFIX}thread-id`);
    return stored || newThreadId();
  } catch {
    // localStorage can throw in private-browsing / storage-disabled contexts
    return newThreadId();
  }
}

function loadMessages(threadId: string): ChatMessage[] {
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}messages:${threadId}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m): m is ChatMessage =>
        m && typeof m.id === "string" && typeof m.content === "string" && typeof m.role === "string"
    );
  } catch {
    return [];
  }
}

function persist(threadId: string, messages: ChatMessage[]) {
  try {
    window.localStorage.setItem(`${STORAGE_PREFIX}thread-id`, threadId);
    window.localStorage.setItem(`${STORAGE_PREFIX}messages:${threadId}`, JSON.stringify(messages));
  } catch {
    // Storage full or unavailable — the conversation still works, it just
    // won't survive a refresh. Not fatal, so we swallow it.
  }
}

export function useFinancialAdvisor() {
  const [mounted, setMounted] = useState(false);
  const [threadId, setThreadId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("checking");

  const abortRef = useRef<AbortController | null>(null);
  const pendingQuestionRef = useRef<string | null>(null);

  // --- hydrate from localStorage exactly once, client-side only ---------
  useEffect(() => {
    const id = loadThreadId();
    setThreadId(id);
    setMessages(loadMessages(id));
    setMounted(true);
  }, []);

  // --- persist on change --------------------------------------------------
  useEffect(() => {
    if (!mounted || !threadId) return;
    persist(threadId, messages);
  }, [mounted, threadId, messages]);

  // --- health polling, with a tighter loop while offline ------------------
  useEffect(() => {
    if (!mounted) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      const controller = new AbortController();
      const ok = await checkAdvisorHealth(controller.signal);
      if (cancelled) return;
      setConnection(ok ? "online" : "offline");
      timer = setTimeout(poll, ok ? HEALTH_POLL_MS : HEALTH_RETRY_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [mounted]);

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const runQuestion = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isThinking) return;

      pendingQuestionRef.current = trimmed;
      appendMessage({
        id: newMessageId(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      });

      const controller = new AbortController();
      abortRef.current = controller;
      setIsThinking(true);

      try {
        const responseText = await askAdvisor({
          question: trimmed,
          threadId,
          signal: controller.signal,
        });
        pendingQuestionRef.current = null;
        appendMessage({
          id: newMessageId(),
          role: "assistant",
          content: responseText,
          createdAt: Date.now(),
        });
      } catch (err) {
        if (err instanceof AdvisorApiError && err.kind === "aborted") {
          // Silent, deliberate user cancellation — nothing to show.
        } else {
          const message =
            err instanceof AdvisorApiError
              ? err.message
              : "Something unexpected went wrong talking to Shankh.";
          appendMessage({
            id: newMessageId(),
            role: "error",
            content: message,
            createdAt: Date.now(),
            retryQuestion: trimmed,
          });
        }
      } finally {
        setIsThinking(false);
        abortRef.current = null;
      }
    },
    [appendMessage, isThinking, threadId]
  );

  const sendMessage = useCallback((question: string) => void runQuestion(question), [runQuestion]);

  const retry = useCallback(
    (question: string) => {
      // Drop the error bubble being retried so the transcript doesn't
      // accumulate duplicate failures for the same question.
      setMessages((prev) => {
        const idx = [...prev].reverse().findIndex((m) => m.role === "error" && m.retryQuestion === question);
        if (idx === -1) return prev;
        const realIdx = prev.length - 1 - idx;
        return [...prev.slice(0, realIdx), ...prev.slice(realIdx + 1)];
      });
      void runQuestion(question);
    },
    [runQuestion]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const startNewThread = useCallback(() => {
    abortRef.current?.abort();
    const id = newThreadId();
    setThreadId(id);
    setMessages(loadMessages(id));
    setIsThinking(false);
  }, []);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  return {
    mounted,
    threadId,
    messages,
    isThinking,
    connection,
    sendMessage,
    retry,
    cancel,
    startNewThread,
  };
}