<script lang="ts">
  import Sidebar from "./Sidebar.svelte";
  import ChatArea from "./ChatArea.svelte";
  import type { ChatMessageInput as ApiChatMessage } from '../services/api';
  import { createEventDispatcher } from 'svelte';

  export let messages: ApiChatMessage[] = [];
  export let currentSessionId: string; // Changed from sessionId to currentSessionId to match App.svelte

  const dispatch = createEventDispatcher();

  // Event handlers to bubble up events from ChatArea to App.svelte
  function handleNewMessages(event: CustomEvent<ApiChatMessage[]>) {
    dispatch('newmessages', event.detail);
  }

  function handleApiError(event: CustomEvent<{ message: string }>) {
    dispatch('apierror', event.detail);
  }
</script>

<div class="layout">
  <!-- Sidebar doesn't directly need messages or full sessionId propagation for chat messages,
       but it uses the sessionId store internally for its own API calls (settings, files) -->
  <Sidebar />

  <!-- Pass messages and currentSessionId to ChatArea -->
  <ChatArea
    messages={messages}
    sessionId={currentSessionId}
    on:newmessages={handleNewMessages}
    on:apierror={handleApiError}
  />
</div>

<style>
  .layout {
    display: flex;
    height: 100vh;
    width: 100vw; /* Ensure it takes full viewport width */
    overflow: hidden; /* Prevent layout itself from scrolling */
  }
</style>
