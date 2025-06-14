<script lang="ts">
  import Layout from "./lib/components/Layout.svelte";
  import "./app.css";
  import { sessionId } from "./lib/stores/sessionStore";
  import type { ChatMessageInput as ApiChatMessage } from './lib/services/api';
  import { onMount } from "svelte";

  let currentSessionId: string;
  let messages: ApiChatMessage[] = [];

  // Subscribe to session ID changes
  sessionId.subscribe(value => {
    if (currentSessionId && currentSessionId !== value) {
      // Session ID has changed (e.g., "New Chat" was clicked)
      console.log("App.svelte: Session ID changed, clearing messages.");
      messages = []; // Clear messages for the new session
    }
    currentSessionId = value;
  });

  function handleNewMessages(event: CustomEvent<ApiChatMessage[]>) {
    messages = event.detail;
    // console.log("App.svelte: newmessages received", messages);
  }

  function handleApiError(event: CustomEvent<{ message: string }>) {
    // Display a more prominent error, perhaps a toast notification in a real app
    console.error("App.svelte: API Error bubbled up:", event.detail.message);
    // Add error to messages list to make it visible in chat
    messages = [
      ...messages,
      {
        role: "assistant",
        content: `ERROR: ${event.detail.message}. Check console for details.`
      }
    ];
  }

  onMount(() => {
    // You could potentially load initial messages for a session here if needed,
    // or fetch initial settings that might affect App.svelte directly.
    // For now, sessionStore handles initial ID, and Sidebar handles settings loading.
    // The first message from the user will initiate the chat.
  });

</script>

<main>
  <!--
    Layout now needs to accept messages and sessionId props if ChatArea is nested within it
    and needs these props.
    Alternatively, if Layout doesn't directly use them but ChatArea (a child of Layout) does,
    we pass them through Layout.
  -->
  <Layout {messages} {currentSessionId} on:newmessages={handleNewMessages} on:apierror={handleApiError} />
</main>

<style>
  main {
    height: 100%;
    width: 100%;
  }
</style>
