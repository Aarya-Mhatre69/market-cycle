"use client";

import * as React from "react";
import {
  Loader2,
  Plus,
  RotateCcw,
  Send,
  Square,
  Sparkles,
  Wifi,
  WifiOff,
  Clock3,
} from "lucide-react";

import { useFinancialAdvisor } from "@/hooks/useFinancialAdvisor";
import type { ChatMessage } from "@/lib/types";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(ts: number) {
  return new Date(ts).toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

function ConnectionBadge({
  state,
}: {
  state: "checking" | "online" | "offline";
}) {
  if (state === "online") {
    return (<Badge variant="secondary" className="gap-1.5 rounded-full border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"> <Wifi className="h-3.5 w-3.5" />
      Online </Badge>
    );
  }

  if (state === "offline") {
    return (<Badge variant="secondary" className="gap-1.5 rounded-full border-rose-500/20 bg-rose-500/10 text-rose-700 dark:text-rose-300"> <WifiOff className="h-3.5 w-3.5" />
      Offline </Badge>
    );
  }

  return (<Badge variant="secondary" className="gap-1.5 rounded-full border-slate-500/20 bg-slate-500/10 text-slate-700 dark:text-slate-300"> <Loader2 className="h-3.5 w-3.5 animate-spin" />
    Checking </Badge>
  );
}

function RoleBadge({ role }: { role: ChatMessage["role"] }) {
  if (role === "user") {
    return (<Badge variant="outline" className="rounded-full">
      You </Badge>
    );
  }

  if (role === "assistant") {
    return (<Badge variant="outline" className="rounded-full border-sky-500/30 bg-sky-500/5 text-sky-700 dark:text-sky-300">
      Shankh </Badge>
    );
  }

  return (<Badge variant="outline" className="rounded-full border-rose-500/30 bg-rose-500/5 text-rose-700 dark:text-rose-300">
    Error </Badge>
  );
}

function MessageCard({
  message,
  onRetry,
}: {
  message: ChatMessage;
  onRetry: (question: string) => void;
}) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const isError = message.role === "error";

  return (
    <Card
      className={[
        "border shadow-sm transition-all",
        isUser ? "ml-auto max-w-[88%] border-sky-500/20 bg-sky-500/5" : "",
        isAssistant ? "mr-auto max-w-[88%] border-border bg-background" : "",
        isError ? "mr-auto max-w-[88%] border-rose-500/20 bg-rose-500/5" : "",
      ].join(" ")}
    > <CardHeader className="space-y-3 pb-3"> <div className="flex items-center justify-between gap-3"> <RoleBadge role={message.role} /> <div className="flex items-center gap-1.5 text-xs text-muted-foreground"> <Clock3 className="h-3.5 w-3.5" />
      {formatDate(message.createdAt)} · {formatTime(message.createdAt)} </div> </div> </CardHeader>

      ```
      <CardContent className="pt-0">
        <div className="whitespace-pre-wrap break-words text-sm leading-6 text-foreground">
          {message.content}
        </div>
      </CardContent>

      {isError && message.retryQuestion ? (
        <CardFooter className="pt-0">
          <Button
            variant="outline"
            size="sm"
            className="gap-2 rounded-full"
            onClick={() => onRetry(message.retryQuestion!)}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Retry
          </Button>
        </CardFooter>
      ) : null}
    </Card>

);
}

function EmptyTranscript() {
return ( <div className="flex h-full min-h-[420px] items-center justify-center px-4"> <Card className="w-full max-w-xl border-dashed bg-muted/30"> <CardHeader className="text-center"> <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-2xl border bg-background"> <Sparkles className="h-5 w-5 text-sky-500" /> </div> <CardTitle>Shankh Advisor</CardTitle> <CardDescription>
Ask about market regime, strategy logic, risk checks, or agent behavior. </CardDescription> </CardHeader> </Card> </div>
);
}

export default function ShankhAdvisorUI() {
const {
mounted,
threadId,
messages,
isThinking,
connection,
sendMessage,
retry,
cancel,
startNewThread,
} = useFinancialAdvisor();

const [input, setInput] = React.useState("");
const scrollRef = React.useRef<HTMLDivElement | null>(null);
const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

React.useEffect(() => {
const el = scrollRef.current;
if (!el) return;
el.scrollTop = el.scrollHeight;
}, [messages, isThinking]);

const quickPrompts = React.useMemo(
() => [
"Explain the current market regime.",
"Summarize this strategy in plain English.",
"What are the main risks?",
],
[]
);

const onSubmit = (e: React.FormEvent) => {
e.preventDefault();
const trimmed = input.trim();
if (!trimmed || isThinking) return;
setInput("");
sendMessage(trimmed);
};

const onQuickPrompt = (prompt: string) => {
setInput(prompt);
textareaRef.current?.focus();
};

return ( <div className="flex h-screen w-full flex-col bg-gradient-to-br from-slate-50 via-background to-slate-100 dark:from-slate-950 dark:via-background dark:to-slate-950"> <header className="border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60"> <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 px-4 py-4"> <div className="min-w-0"> <div className="flex items-center gap-2"> <div className="flex h-9 w-9 items-center justify-center rounded-xl border bg-background shadow-sm"> <Sparkles className="h-4 w-4 text-sky-500" /> </div> <div className="min-w-0"> <h1 className="truncate text-base font-semibold tracking-tight">
Shankh Advisor </h1> <p className="truncate text-sm text-muted-foreground">
{threadId ? `Thread ${ threadId } ` : "Initializing conversation…"} </p> </div> </div> </div>

```
    < div className = "flex flex-wrap items-center justify-end gap-2" >
        <ConnectionBadge state={connection} />
        <Button variant="outline" size="sm" onClick={startNewThread} className="gap-2 rounded-full">
          <Plus className="h-4 w-4" />
          New thread
        </Button>
  {
    isThinking ? (
      <Button variant="destructive" size="sm" onClick={cancel} className="gap-2 rounded-full">
        <Square className="h-4 w-4" />
        Stop
      </Button>
    ) : null
  }
      </div >
    </div >
  </header >

    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 px-4 py-4">
      <Card className="border bg-background shadow-sm">
        <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
          <div>
            <p className="text-sm font-medium">Quick prompts</p>
            <p className="text-xs text-muted-foreground">
              Start with a strategy, explanation, or risk question.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {quickPrompts.map((prompt) => (
              <Button
                key={prompt}
                type="button"
                variant="secondary"
                size="sm"
                className="rounded-full"
                disabled={isThinking}
                onClick={() => onQuickPrompt(prompt)}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-1 flex-col overflow-hidden border shadow-sm">
        <CardHeader className="border-b pb-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">Conversation</CardTitle>
              <CardDescription>
                Messages are stored locally per thread.
              </CardDescription>
            </div>
            {messages.length ? (
              <Badge variant="outline" className="rounded-full">
                {messages.length} messages
              </Badge>
            ) : null}
          </div>
        </CardHeader>

        <ScrollArea className="flex-1">
          <div ref={scrollRef} className="space-y-4 p-4">
            {!mounted || messages.length === 0 ? (
              <EmptyTranscript />
            ) : (
              messages.map((message) => (
                <MessageCard key={message.id} message={message} onRetry={retry} />
              ))
            )}

            {isThinking ? (
              <Card className="mr-auto max-w-[88%] border-border bg-muted/40">
                <CardContent className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Thinking…
                </CardContent>
              </Card>
            ) : null}
          </div>
        </ScrollArea>

        <Separator />

        <CardFooter className="p-4">
          <form onSubmit={onSubmit} className="w-full">
            <div className="flex flex-col gap-3">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Shankh anything…"
                className="min-h-28 resize-none"
                disabled={isThinking}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit(e as unknown as React.FormEvent);
                  }
                }}
              />

              <div className="flex items-center justify-between gap-3">
                <p className="text-xs text-muted-foreground">
                  Enter to send · Shift+Enter for a new line
                </p>

                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="gap-2 rounded-full"
                    onClick={() => setInput("")}
                    disabled={!input.trim() || isThinking}
                  >
                    Clear
                  </Button>
                  <Button
                    type="submit"
                    className="gap-2 rounded-full"
                    disabled={!input.trim() || isThinking}
                  >
                    {isThinking ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Sending
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4" />
                        Send
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </form>
        </CardFooter>
      </Card>
    </main>
</div >

);
}
