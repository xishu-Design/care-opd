import { API_BASE_URL } from "../app/constants";
import { useAuthStore } from "../stores/authStore";

export type ChatStreamEvent<TResult> =
  | { type: "status"; message: string }
  | { type: "answer"; text: string }
  | { type: "complete"; result: TResult }
  | { type: "error"; message: string };

export async function postChatStream<TResult>(
  path: string,
  message: string,
  onEvent: (event: ChatStreamEvent<TResult>) => void,
  signal?: AbortSignal,
) {
  const token = useAuthStore.getState().token;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error("无法连接流式问答服务。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const eventType = block.match(/^event: (.+)$/m)?.[1];
      const rawData = block.match(/^data: (.+)$/m)?.[1];
      if (!eventType || !rawData) continue;
      const data = JSON.parse(rawData) as Record<string, unknown>;
      if (eventType === "status" && typeof data.message === "string") {
        onEvent({ type: "status", message: data.message });
      }
      if (eventType === "answer" && typeof data.text === "string") {
        await emitCharacters(data.text, onEvent, signal);
      }
      if (eventType === "error" && typeof data.message === "string") {
        onEvent({ type: "error", message: data.message });
      }
      if (eventType === "complete" && data.result && typeof data.result === "object") {
        onEvent({ type: "complete", result: data.result as TResult });
      }
    }
    if (done) break;
  }
}

async function emitCharacters<TResult>(
  text: string,
  onEvent: (event: ChatStreamEvent<TResult>) => void,
  signal?: AbortSignal,
) {
  for (const character of text) {
    if (signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError");
    onEvent({ type: "answer", text: character });
    await abortableDelay(10, signal);
  }
}

function abortableDelay(milliseconds: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("The operation was aborted.", "AbortError"));
      return;
    }
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("The operation was aborted.", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
