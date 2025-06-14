<script lang="ts">
  export let role: 'user' | 'assistant';
  export let content: string;

  // Placeholder for future props related to special content
  // export let fileUrl: string | undefined = undefined;
  // export let ragSources: Array<{ type: 'pdf' | 'web'; name: string; url: string }> | undefined = undefined;
  // export let codeOutput: { type: 'image' | 'download'; content: string; filename?: string } | undefined = undefined;

  const isUser = role === 'user';
</script>

<div class="chat-message" class:user={isUser} class:assistant={!isUser}>
  <div class="message-bubble">
    <p>{content}</p>

    <!-- Placeholder for RAG sources -->
    {#if role === 'assistant' /* && ragSources && ragSources.length > 0 */}
      <div class="rag-sources-placeholder">
        <!--
          RAG sources rendering will go here.
          Example:
          {#each ragSources as source}
            <a href={source.url} target="_blank">[{source.type.toUpperCase()}] {source.name}</a>
          {/each}
        -->
        <p style="font-size: 0.8em; color: #555;"><em>[RAG sources will appear here]</em></p>
      </div>
    {/if}

    <!-- Placeholder for file downloads (from assistant) -->
    {#if role === 'assistant' /* && fileUrl */}
      <div class="file-download-placeholder">
        <!--
          File download link/button will go here.
          Example: <a href={fileUrl} download>Download file</a>
        -->
        <p style="font-size: 0.8em; color: #555;"><em>[File download link will appear here]</em></p>
      </div>
    {/if}

    <!-- Placeholder for code interpreter output -->
    {#if role === 'assistant' /* && codeOutput */}
      <div class="code-output-placeholder">
        <!--
          Code interpreter output (e.g., image or download button for a file) will go here.
          Example for image: <img src={codeOutput.content} alt="Code output" />
          Example for download: <a href={codeOutput.content} download={codeOutput.filename}>Download {codeOutput.filename}</a>
        -->
        <p style="font-size: 0.8em; color: #555;"><em>[Code interpreter output will appear here]</em></p>
      </div>
    {/if}
  </div>
</div>

<style>
  .chat-message {
    display: flex;
    margin-bottom: 0.75rem;
    max-width: 80%;
  }

  .message-bubble {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    word-wrap: break-word; /* Ensure long words don't overflow */
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  }

  .chat-message.user {
    margin-left: auto; /* Align user messages to the right */
    flex-direction: row-reverse;
  }

  .chat-message.user .message-bubble {
    background-color: #007bff; /* Blue for user */
    color: white;
    border-bottom-right-radius: 4px; /* Slightly different shape for user */
  }

  .chat-message.assistant {
    margin-right: auto; /* Align assistant messages to the left */
  }

  .chat-message.assistant .message-bubble {
    background-color: #e9ecef; /* Light gray for assistant */
    color: #333;
    border-bottom-left-radius: 4px; /* Slightly different shape for assistant */
  }

  /* Styles for placeholder sections */
  .rag-sources-placeholder,
  .file-download-placeholder,
  .code-output-placeholder {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(0,0,0,0.05);
  }
</style>
