import type {
  PublicCorrectionRequest,
  PublicCorrectionResponse,
} from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type APIErrorPayload = {
  error?: string;
  detail?: string | Array<{ msg?: string }>;
};

function getErrorMessage(payload: APIErrorPayload): string {
  if (payload.error) {
    return payload.error;
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    const message = payload.detail.find((item) => item.msg)?.msg;
    if (message) {
      return message.replace(/^Value error,\s*/i, "");
    }
  }
  return "We could not correct your text. Please try again.";
}

export async function correctText(
  text: string,
): Promise<PublicCorrectionResponse> {
  const request: PublicCorrectionRequest = { text };
  const response = await fetch(`${API_URL}/public/correct`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let payload: APIErrorPayload = {};
    try {
      payload = (await response.json()) as APIErrorPayload;
    } catch {
      // Keep the fallback message when the server does not return JSON.
    }
    throw new Error(getErrorMessage(payload));
  }

  return (await response.json()) as PublicCorrectionResponse;
}

export async function checkAPIHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`, {
      method: "GET",
      cache: "no-store",
    });

    if (!response.ok) {
      return false;
    }

    const payload = (await response.json()) as { status?: string };
    return payload.status === "ok";
  } catch {
    return false;
  }
}
