<script lang="ts">
  import { onMount } from 'svelte';
  import { sessionId, resetSessionId as resetGlobalSessionId, getSessionId as getCurrentSessionId } from '../stores/sessionStore';
  import { saveLLMSettings, getLLMSettings, uploadFile as apiUploadFile } from '../services/api';
  import type { LLMSettingsInput, FileUploadResponse } from '../services/api';

  // LLM Settings
  let temperature: number = 0.7;
  let verbosity: number = 3;
  let numSearchResults: number = 10;
  let longTermMemoryEnabled: boolean = true; // Note: Backend doesn't handle this setting yet

  let currentSessionId: string;
  sessionId.subscribe(value => {
    currentSessionId = value;
    // Fetch settings when session ID changes or on initial load if currentSessionId is already available
    if (currentSessionId) {
      loadSettingsForSession(currentSessionId);
    }
  });

  async function loadSettingsForSession(sid: string) {
    console.log("Sidebar: Attempting to load settings for session", sid);
    try {
      const settings = await getLLMSettings(sid);
      temperature = settings.temperature ?? 0.7;
      verbosity = settings.verbosity ?? 3;
      numSearchResults = settings.max_search_results ?? 10;
      // longTermMemoryEnabled = settings.long_term_memory ?? true; // When backend supports
      console.log("Sidebar: Settings loaded for session", sid, settings);
    } catch (error) {
      console.error("Sidebar: Failed to load LLM settings for session", sid, error);
      // Keep current UI values or reset to defaults if preferred
    }
  }

  async function handleSettingChange() {
    if (!currentSessionId) return;
    const settingsToSave: LLMSettingsInput = {
      temperature,
      verbosity,
      max_search_results: numSearchResults,
      // long_term_memory: longTermMemoryEnabled, // When backend supports
    };
    try {
      await saveLLMSettings(currentSessionId, settingsToSave);
      console.log("Sidebar: Settings saved for session", currentSessionId);
    } catch (error) {
      console.error("Sidebar: Failed to save LLM settings for session", currentSessionId, error);
      // Optionally show an error to the user
    }
  }

  // File Uploads
  interface UploadedFileDisplay {
    name: string;
    status: 'uploading' | 'success' | 'error';
    message?: string; // For error messages
    serverPath?: string; // Store path from server for reference
  }
  let displayedFiles: UploadedFileDisplay[] = [];
  let fileInput: HTMLInputElement;

  async function handleFileUpload(event: Event) {
    if (!currentSessionId) {
      alert("Session ID is not available. Cannot upload file.");
      return;
    }
    const target = event.target as HTMLInputElement;
    if (target.files) {
      const filesToUpload = Array.from(target.files);

      for (const file of filesToUpload) {
        // Add to display list with 'uploading' status
        displayedFiles = [...displayedFiles, { name: file.name, status: 'uploading' }];

        try {
          const response: FileUploadResponse = await apiUploadFile(currentSessionId, file);
          // Update status to 'success'
          displayedFiles = displayedFiles.map(f =>
            f.name === file.name && f.status === 'uploading'
            ? { ...f, status: 'success', serverPath: response.filepath }
            : f
          );
          console.log("Sidebar: File uploaded", response);
        } catch (error: any) {
          console.error("Sidebar: File upload error", error);
          // Update status to 'error'
          displayedFiles = displayedFiles.map(f =>
            f.name === file.name && f.status === 'uploading'
            ? { ...f, status: 'error', message: error.message || "Upload failed" }
            : f
          );
        }
      }
      target.value = ""; // Clear the input after processing
    }
  }

  function removeFileFromDisplay(fileName: string) {
    displayedFiles = displayedFiles.filter(file => file.name !== fileName);
    // Note: This only removes from UI. No call to delete from server yet.
  }

  // Chat History (placeholders)
  const hardcodedChats = [
    // { id: "1", name: "Svelte Discussion" }, // Example, will be replaced by dynamic logic
  ];

  function newChat() {
    console.log("New Chat button clicked in Sidebar");
    resetGlobalSessionId(); // This generates a new session ID and updates the store
    // Parent component (App.svelte or Layout.svelte) should listen to sessionId changes
    // and clear chat messages in ChatArea.svelte.
    // For now, just clearing displayed files in sidebar for the new session.
    displayedFiles = [];
    // LLM Settings will be fetched for the new session ID due to the subscription
  }

  function renameChat(id: string) {
    console.log("Rename chat:", id);
    // Placeholder - actual implementation requires backend
  }

  function deleteChat(id: string) {
    console.log("Delete chat:", id);
    // Placeholder - actual implementation requires backend
  }

  onMount(() => {
    // Initial load of settings for the current session ID
    // The subscription to sessionId store already handles this.
    // If currentSessionId is available onMount, loadSettingsForSession will be called by subscription.
    // If not, it will be called once sessionId store updates with a valid ID.
  });

</script>

<div class="sidebar">
  <div class="sidebar-content">
    <details class="sidebar-section" open>
      <summary>Chat History</summary>
      <div class="section-content">
        <button class="sidebar-button full-width" on:click={newChat}>+ New Chat</button>
        <ul class="chat-list">
          {#each hardcodedChats as chat (chat.id)}
            <li class="chat-list-item">
              <span>{chat.name}</span>
              <div class="chat-actions">
                <button class="icon-button" title="Rename Chat" on:click={() => renameChat(chat.id)}>✏️</button>
                <button class="icon-button" title="Delete Chat" on:click={() => deleteChat(chat.id)}>🗑️</button>
              </div>
            </li>
          {/each}
          {#if hardcodedChats.length === 0}
            <p class="empty-list-placeholder">Chat history not yet implemented.</p>
          {/if}
        </ul>
      </div>
    </details>

    <details class="sidebar-section" open>
      <summary>Upload Files</summary>
      <div class="section-content">
        <input
          type="file"
          bind:this={fileInput}
          on:change={handleFileUpload}
          multiple
          aria-label="File uploader"
          class="file-input"
          disabled={!currentSessionId}
        />
        {#if displayedFiles.length > 0}
          <ul class="file-list">
            {#each displayedFiles as file (file.name + file.status)} {/* Add status to key for reactivity */}
              <li class="file-list-item" class:success={file.status === 'success'} class:error={file.status === 'error'} class:uploading={file.status === 'uploading'}>
                <span class="file-name" title={file.name}>{file.name}</span>
                {#if file.status === 'uploading'}
                  <span class="status-badge">Uploading...</span>
                {:else if file.status === 'success'}
                  <span class="status-badge success">✓</span>
                {:else if file.status === 'error'}
                  <span class="status-badge error" title={file.message}>✗ Failed</span>
                {/if}
                <button class="icon-button remove-file" title="Remove file" on:click={() => removeFileFromDisplay(file.name)} disabled={file.status === 'uploading'}>✕</button>
              </li>
            {/each}
          </ul>
        {/if}
        {#if displayedFiles.length === 0}
          <p class="empty-list-placeholder">No files uploaded for this session.</p>
        {/if}
      </div>
    </details>

    <details class="sidebar-section" open>
      <summary>LLM Settings</summary>
      <div class="section-content">
        <div class="setting">
          <label for="temperature">Creativity (Temperature): {temperature.toFixed(2)}</label>
          <input type="range" id="temperature" bind:value={temperature} min="0" max="1" step="0.01" on:change={handleSettingChange} disabled={!currentSessionId} />
        </div>
        <div class="setting">
          <label for="verbosity">Verbosity: {verbosity}</label>
          <input type="range" id="verbosity" bind:value={verbosity} min="0" max="5" step="1" on:change={handleSettingChange} disabled={!currentSessionId} />
        </div>
        <div class="setting">
          <label for="numSearchResults">Search Results: {numSearchResults}</label>
          <input type="range" id="numSearchResults" bind:value={numSearchResults} min="1" max="20" step="1" on:change={handleSettingChange} disabled={!currentSessionId} />
        </div>
        <div class="setting toggle-setting">
          <label for="longTermMemory">Enable Long-term Memory:</label>
          <label class="switch">
            <input type="checkbox" id="longTermMemory" bind:checked={longTermMemoryEnabled} on:change={handleSettingChange} disabled={!currentSessionId || true} /> {/* LTM setting disabled as backend doesn't use it yet */}
            <span class="slider round"></span>
          </label>
        </div>
         {#if !currentSessionId}
          <p class="empty-list-placeholder notice">Session ID not available. Settings disabled.</p>
        {/if}
      </div>
    </details>

    <details class="sidebar-section">
      <summary>About ESI</summary>
      <div class="section-content about-esi">
        <p><strong>ESI: Your Enterprise Strategy Intelligence Agent</strong></p>
        <p><em>This is a prototype application.</em></p>
        <p><strong>Data Privacy & Security:</strong> Your data, including uploaded documents and chat history, is processed to provide the service. If long-term memory is enabled, chat data is stored. Ensure compliance with your organization's data policies before uploading sensitive information.</p>
        <p><strong>Limitations:</strong> The AI may occasionally produce incorrect or misleading information. Always critically evaluate its outputs.</p>
        <p><strong>Feedback:</strong> Your feedback is valuable for improving ESI. Contact the development team with any issues or suggestions.</p>
      </div>
    </details>
  </div>
</div>

<style>
  .sidebar {
    width: 300px;
    background-color: #f0f4f8;
    padding: 0;
    border-right: 1px solid #d1d9e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-y: hidden;
  }

  .sidebar-content {
    padding: 0.75rem;
    overflow-y: auto;
    height: 100%;
  }

  .sidebar-section {
    margin-bottom: 0.75rem;
    border: 1px solid #c5d0db;
    border-radius: 6px;
    background-color: #fff;
  }

  .sidebar-section summary {
    font-weight: 600;
    padding: 0.75rem;
    cursor: pointer;
    background-color: #e9edf2;
    border-radius: 6px 6px 0 0;
    border-bottom: 1px solid #c5d0db;
    outline: none;
    transition: background-color 0.2s ease;
  }
  .sidebar-section summary:hover {
    background-color: #dde4eb;
  }

  .sidebar-section[open] summary {
    border-bottom: 1px solid #c5d0db;
  }

  .section-content {
    padding: 0.75rem;
  }

  .sidebar-button.full-width {
    width: 100%;
    padding: 0.6rem;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-align: center;
    margin-bottom: 0.75rem;
    transition: background-color 0.2s ease;
  }
  .sidebar-button.full-width:hover {
    background-color: #0056b3;
  }

  .chat-list, .file-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .chat-list-item, .file-list-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.25rem;
    border-bottom: 1px solid #e9ecef;
    font-size: 0.9em;
  }
  .file-list-item.success { background-color: #e6ffed; }
  .file-list-item.error { background-color: #ffe6e6; }
  .file-list-item.uploading { background-color: #f0f0f0; }

  .chat-list-item:last-child, .file-list-item:last-child {
    border-bottom: none;
  }

  .file-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 150px; /* Adjusted for status badge */
  }

  .icon-button {
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.25rem;
    font-size: 0.9em;
    margin-left: 0.25rem;
  }
  .icon-button:hover {
    opacity: 0.7;
  }
  .remove-file {
    color: #dc3545;
    font-weight: bold;
  }
  .remove-file:disabled {
    color: #adb5bd;
    cursor: not-allowed;
  }


  .file-input {
    width: 100%;
    margin-bottom: 0.75rem;
    font-size: 0.9em;
  }
  .file-input:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  .empty-list-placeholder {
    font-size: 0.85em;
    color: #6c757d;
    text-align: center;
    padding: 0.5rem 0;
  }
  .empty-list-placeholder.notice {
    font-style: italic;
    font-size: 0.8em;
  }

  .setting {
    margin-bottom: 1rem;
  }
  .setting label {
    display: block;
    margin-bottom: 0.3rem;
    font-size: 0.9em;
    color: #333;
  }
  .setting input[type="range"] {
    width: 100%;
  }
  .setting input[type="range"]:disabled {
    opacity: 0.5;
  }


  .toggle-setting {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .toggle-setting input[type="checkbox"]:disabled + .slider {
    opacity: 0.5;
    cursor: not-allowed;
  }


  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }
  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
  }
  .slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: .4s;
  }
  input:checked + .slider {
    background-color: #007bff;
  }
  input:focus + .slider {
    box-shadow: 0 0 1px #007bff;
  }
  input:checked + .slider:before {
    transform: translateX(20px);
  }
  .slider.round {
    border-radius: 24px;
  }
  .slider.round:before {
    border-radius: 50%;
  }

  .about-esi p {
    font-size: 0.85em;
    line-height: 1.5;
    margin-bottom: 0.6rem;
    color: #495057;
  }
  .about-esi strong {
    color: #343a40;
  }
  .about-esi em {
    color: #007bff;
  }

  .status-badge {
    font-size: 0.8em;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    margin-left: 0.5em;
  }
  .status-badge.success { color: green; }
  .status-badge.error { color: red; }
  .status-badge.uploading { color: #555; }
</style>
