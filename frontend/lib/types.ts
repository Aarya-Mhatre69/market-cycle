export type MessageRole = "user" | "assistant" | "error";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  /** Present on role "error" — lets the composer offer a one-click retry. */
  retryQuestion?: string;
}

export type ConnectionState = "checking" | "online" | "offline";

export type AdvisorErrorKind =
  | "network"
  | "timeout"
  | "aborted"
  | "unavailable"
  | "server"
  | "invalid_response";

export interface AskAdvisorPayload {
  question: string;
  thread_id: string;
}

export interface AskAdvisorResponse {
  response: string;
}