<script lang="ts">
  import ChatMessage from "./ChatMessage.svelte";
  import ChatInput from "./ChatInput.svelte";
  import type { ChatMessageInput as ApiChatMessage } from '../services/api';
  import { onMount, afterUpdate, createEventDispatcher }_from 'svelte';

  export let messages: ApiChatMessage[] = []; // Prop: messages are managed by a parent
  export let sessionId: string; // Prop: session ID managed by a parent or store

  const dispatch = createEventDispatcher();

  let messagesContainer: HTMLElement;
  let autoScrollEnabled = true;

  // Event handlers are now for events bubbled up from ChatInput
  function handleNewMessages(event: CustomEvent<ApiChatMessage[]>) {
    // Bubble up to parent (App.svelte)
    dispatch('newmessages', event.detail);
  }

  function handleApiError(event: CustomEvent<{ message: string }>) {
    // Bubble up to parent (App.svelte)
    console.error("ChatArea received API Error:", event.detail.message);
    dispatch('apierror', event.detail);
  }

  function scrollToBottom() {
    if (messagesContainer && autoScrollEnabled) {
      messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: 'smooth' });
    }
  }

  function handleScroll() {
    if (!messagesContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
    if (scrollHeight - scrollTop - clientHeight > 100) {
      autoScrollEnabled = false;
    } else {
      autoScrollEnabled = true;
    }
  }

  onMount(() => {
    if (messagesContainer) {
      messagesContainer.addEventListener('scroll', handleScroll);
    }
    scrollToBottom();
  });

  afterUpdate(() => {
    scrollToBottom();
  });

</script>

<div class="chat-area-container">
  <div class="messages" bind:this={messagesContainer}>
    {#if !sessionId}
      <p class="session-notice">Initializing session...</p>
    {:else if messages.length === 0}
      <p class="session-notice">No messages yet. Start by typing below.</p>
    {/if}
    {#each messages as message, i (i)} {/* Using index as key; consider unique IDs if available */}
      <ChatMessage role={message.role} content={message.content} />
    {/each}
  </div>

  <div class="suggested-prompts-placeholder">
    <button class="prompt-button">Suggest: "Explain Svelte Stores"</button>
    <button class="prompt-button">Suggest: "How to deploy a Svelte app?"</button>
    <button class="prompt-button">Suggest: "Compare Svelte with React"</button>
  </div>

  <ChatInput {messages} {sessionId} on:newmessages={handleNewMessages} on:apierror={handleApiError} />
</div>

<style>
  .chat-area-container {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    background-color: #fff;
    overflow: hidden;
    padding: 0;
  }

  .messages {
    flex-grow: 1;
    padding: 1rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid #e0e0e0;
  }

  .session-notice {
    text-align: center;
    color: #6c757d;
    padding: 1rem;
    font-style: italic;
  }

  .suggested-prompts-placeholder {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    background-color: #f8f9fa;
  }

  .prompt-button {
    padding: 0.5rem 0.75rem;
    font-size: 0.85em;
    background-color: #e9ecef;
    border: 1px solid #ced4da;
    border-radius: 15px;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .prompt-button:hover {
    background-color: #dee2e6;
  }
</style>
