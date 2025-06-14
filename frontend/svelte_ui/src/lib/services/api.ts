// Define simple types for chat messages, aligning with backend Pydantic models
export interface ChatMessageInput {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatApiResponse {
  assistant_response: string;
  updated_chat_history: ChatMessageInput[];
  session_id: string; // Expect session_id back from chat endpoint
}

export interface LLMSettingsInput {
  temperature?: number;
  verbosity?: number;
  max_search_results?: number;
  // Add long_term_memory when backend supports it for settings
}

export interface LLMSettingsResponse extends LLMSettingsInput {
  session_id: string;
}

export interface FileUploadResponse {
  message: string;
  session_id: string;
  filename: string;
  filepath: string;
}

const BASE_API_URL = 'http://localhost:8000/api'; // Base URL for all API calls

/**
 * Sends a user query and chat history to the backend API.
 * @param query The user's current message.
 * @param chatHistory The existing array of chat messages.
 * @param sessionId The current session ID.
 * @returns A Promise that resolves to the API response.
 */
export async function sendMessageToApi(
  query: string,
  chatHistory: ChatMessageInput[],
  sessionId: string
): Promise<ChatApiResponse> {
  console.log('Sending to API (/api/chat):', { query, chatHistory, sessionId });
  try {
    const response = await fetch(`${BASE_API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        chat_history: chatHistory,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error('API Error Response from /chat:', errorBody);
      throw new Error(`API request to /chat failed with status ${response.status}: ${errorBody}`);
    }

    const data: ChatApiResponse = await response.json();
    console.log('Received from API (/api/chat):', data);
    return data;
  } catch (error) {
    console.error('Failed to send message or parse response from /chat:', error);
    throw error;
  }
}

/**
 * Saves LLM settings to the backend.
 * @param sessionId The current session ID.
 * @param settings The LLM settings to save.
 * @returns A Promise that resolves to the updated settings from the API.
 */
export async function saveLLMSettings(
  sessionId: string,
  settings: LLMSettingsInput
): Promise<LLMSettingsResponse> {
  console.log('Saving LLM Settings to API (/api/settings):', { sessionId, settings });
  try {
    const response = await fetch(`${BASE_API_URL}/settings/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error('API Error Response from /settings (POST):', errorBody);
      throw new Error(`API request to /settings (POST) failed with status ${response.status}: ${errorBody}`);
    }

    const data: LLMSettingsResponse = await response.json();
    console.log('Received from API (/settings POST):', data);
    return data;
  } catch (error) {
    console.error('Failed to save LLM settings:', error);
    throw error;
  }
}

/**
 * Gets LLM settings from the backend.
 * @param sessionId The current session ID.
 * @returns A Promise that resolves to the settings from the API.
 */
export async function getLLMSettings(
  sessionId: string
): Promise<LLMSettingsResponse> {
  console.log('Getting LLM Settings from API (/api/settings):', { sessionId });
  try {
    const response = await fetch(`${BASE_API_URL}/settings/${sessionId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error('API Error Response from /settings (GET):', errorBody);
      throw new Error(`API request to /settings (GET) failed with status ${response.status}: ${errorBody}`);
    }

    const data: LLMSettingsResponse = await response.json();
    console.log('Received from API (/settings GET):', data);
    return data;
  } catch (error) {
    console.error('Failed to get LLM settings:', error);
    throw error;
  }
}

/**
 * Uploads a file to the backend.
 * @param sessionId The current session ID.
 * @param file The file to upload.
 * @returns A Promise that resolves to the file upload response from the API.
 */
export async function uploadFile(
  sessionId: string,
  file: File
): Promise<FileUploadResponse> {
  console.log('Uploading file to API (/api/upload_file):', { sessionId, fileName: file.name });
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${BASE_API_URL}/upload_file/${sessionId}`, {
      method: 'POST',
      // 'Content-Type' for FormData is set automatically by the browser
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error('API Error Response from /upload_file:', errorBody);
      throw new Error(`API request to /upload_file failed with status ${response.status}: ${errorBody}`);
    }

    const data: FileUploadResponse = await response.json();
    console.log('Received from API (/upload_file):', data);
    return data;
  } catch (error) {
    console.error('Failed to upload file:', error);
    throw error;
  }
}
