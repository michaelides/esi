import axios from 'axios';

// Smart URL detection for different environments
const getBackendUrl = () => {
  // 1. Use explicit environment variable if set
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }

  // 2. If running in browser, detect environment
  if (typeof window !== 'undefined' && window.location) {
    const origin = window.location.origin;

    // If it's localhost on a non-standard port (likely Vite dev server), use backend port
    if (origin.includes('localhost:') && !origin.includes('localhost:8000')) {
      return 'http://127.0.0.1:8000';
    }

    // If it's not localhost, assume it's a deployment and use the current origin
    if (!origin.includes('localhost')) {
      return origin;
    }
  }

  // 3. Default fallback for local development
  return 'http://127.0.0.1:8000';
};

const BACKEND_URL = getBackendUrl();

// Preferred: send full chat history
export async function runChatWithHistory(messages, options = {}, file = null) {
  try {
    const payload = new FormData();
    if (file) {
      payload.append('file', file);
    }
    payload.append('messages', JSON.stringify(messages));
    payload.append('options', JSON.stringify(options));

    const { data } = await axios.post(`${BACKEND_URL}/chat`, payload);
    return data; // { text, artifacts }
  } catch (error) {
    console.error('Error communicating with the API:', error);
    return { text: "Sorry, I can't complete that request. Please try again." };
  }
}

// Legacy one-shot: wrap single prompt into messages
export default async function runChat(prompt, options = {}) {
  const messages = [{ role: 'user', content: String(prompt ?? '').trim() }];
  return runChatWithHistory(messages, options);
}

export async function streamChatWithHistory(messages, options = {}, file = null, onDelta, signal = null) {
  try {
    const payload = new FormData();
    if (file) {
      payload.append('file', file);
    }
    payload.append('messages', JSON.stringify(messages));
    payload.append('options', JSON.stringify(options));

    const response = await fetch(`${BACKEND_URL}/chat/stream`, {
      method: 'POST',
      body: payload,
      signal: signal // Add abort signal
    });

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      // Check if the request has been aborted
      if (signal && signal.aborted) {
        throw new Error('Request aborted');
      }
      
      const { done, value } = await reader.read();
      if (done) {
        break; // Exit loop when stream is finished
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep the last partial line

      for (const line of lines) {
        // Check if the request has been aborted
        if (signal && signal.aborted) {
          throw new Error('Request aborted');
        }
        
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          if (!jsonStr.trim()) continue;
          try {
            const data = JSON.parse(jsonStr);
            // Pass data to the callback
            if (onDelta) {
              onDelta(data);
            }
            // If server signals 'done', we can stop
            if (data.type === 'done') {
              return;
            }
          } catch (e) {
            console.error('Error parsing stream data:', e, `|${jsonStr}|`);
          }
        }
      }
    }

    // If the loop finishes, the stream has closed without a 'done' event.
    // We need to signal completion to the client.
    if (onDelta) {
      onDelta({ type: 'done' });
    }
  } catch (error) {
    // Handle abort errors specifically
    if (error.name === 'AbortError' || error.message === 'Request aborted') {
      console.log('Request was cancelled');
      throw error; // Re-throw to be handled upstream
    }
    
    console.error('Error communicating with the streaming API:', error);
    if (onDelta) {
      onDelta({ type: 'error', message: "Sorry, I can't complete that request. Please try again." });
    }
  }
}