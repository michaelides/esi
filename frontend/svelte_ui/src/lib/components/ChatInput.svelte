<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import type { ChatMessageInput } from '../services/api';
  import { sendMessageToApi } from '../services/api';

  export let messages: ChatMessageInput[] = []; // Prop to receive current chat history
  export let sessionId: string; // Prop to receive current session ID

  let messageContent: string = "";
  let textareaElement: HTMLTextAreaElement;
  let isSending: boolean = false;

  const dispatch = createEventDispatcher();

  const MAX_TEXTAREA_HEIGHT = 150;

  function adjustTextareaHeight() {
    if (!textareaElement) return;
    textareaElement.style.height = "auto";
    let scrollHeight = textareaElement.scrollHeight;
    if (scrollHeight > MAX_TEXTAREA_HEIGHT) {
      textareaElement.style.height = MAX_TEXTAREA_HEIGHT + "px";
      textareaElement.style.overflowY = "auto";
    } else {
      textareaElement.style.height = scrollHeight + "px";
      textareaElement.style.overflowY = "hidden";
    }
  }

  async function submitMessage() {
    if (messageContent.trim() === "" || isSending) return;
    if (!sessionId) {
      console.error("ChatInput: No session ID available. Cannot send message.");
      dispatch('apierror', { message: 'Session not initialized. Please refresh.' });
      return;
    }

    isSending = true;
    const currentQuery = messageContent;
    messageContent = "";
    textareaElement.style.height = "auto";
    adjustTextareaHeight();

    try {
      const apiResponse = await sendMessageToApi(currentQuery, messages, sessionId);
      dispatch('newmessages', apiResponse.updated_chat_history);
    } catch (error: any) {
      console.error("ChatInput: Error sending message:", error);
      dispatch('apierror', { message: error.message || 'Failed to send message. Please try again.' });
    } finally {
      isSending = false;
      textareaElement.focus();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitMessage();
    }
  }

  onMount(() => {
    adjustTextareaHeight();
    if (textareaElement) { // Ensure textareaElement is available
      textareaElement.focus();
    }
  });
</script>

<div class="chat-input-area">
  <textarea
    bind:this={textareaElement}
    bind:value={messageContent}
    on:input={adjustTextareaHeight}
    on:keydown={handleKeydown}
    placeholder="Type your message (Shift+Enter for newline)..."
    rows="1"
    disabled={isSending || !sessionId}
  ></textarea>
  <button on:click={submitMessage} title="Send message" disabled={isSending || !sessionId}>
    {#if isSending}
      <div class="spinner"></div>
    {:else}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
      </svg>
    {/if}
  </button>
</div>

<style>
  .chat-input-area {
    display: flex;
    align-items: flex-end;
    padding: 0.75rem;
    background-color: #f8f9fa;
    border-top: 1px solid #e0e0e0;
  }

  textarea {
    flex-grow: 1;
    padding: 0.75rem;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 1rem;
    line-height: 1.5;
    resize: none;
    overflow-y: hidden;
    min-height: calc(1.5em + 1.5rem);
    max-height: 150px;
    box-sizing: border-box;
  }

  textarea:disabled {
    background-color: #e9ecef;
  }

  button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem;
    margin-left: 0.75rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    height: calc(1.5em + 1.5rem);
    width: calc(1.5em + 1.5rem);
    transition: background-color 0.2s ease;
  }

  button:hover:not(:disabled) {
    background-color: #0056b3;
  }

  button:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
