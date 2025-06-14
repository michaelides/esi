import { writable } from 'svelte/store';

// Function to generate a unique session ID
function createSessionId() {
  return crypto.randomUUID();
}

// Create a writable store for the session ID
export const sessionId = writable<string>(createSessionId());

// Function to reset the session ID (e.g., for a "New Chat" action)
export function resetSessionId() {
  sessionId.set(createSessionId());
  console.log("New session ID generated:", getSessionId()); // For debugging
}

// Helper to get current session_id value if needed outside component subscription
export function getSessionId(): string {
  let currentSessionId = '';
  sessionId.subscribe(value => {
    currentSessionId = value;
  })(); // Immediately unsubscribe after getting the value
  return currentSessionId;
}

// Log initial session ID
console.log("Initial session ID:", getSessionId());
