import React, { createContext, useState, useEffect, useRef } from "react";  
import runChat, { runChatWithHistory, streamChatWithHistory } from "../config/gemini";
import { marked } from 'marked';
import { supabase } from '../lib/supabaseClient';
import { MarkdownBufferManager } from '../lib/MarkdownBufferManager';
export const Context = createContext();

// Model configuration for LLM selection
const MODEL_CATEGORIES = {
  gemini: {
    label: "Google Gemini",
    models: [
      { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", description: "Fast, balanced performance", maxTemperature: 1.0, pricing: "free" },
      { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", description: "Advanced capabilities", maxTemperature: 1.0, pricing: "paid" },
      { id: "gemini-2.5-flash-lite", name: "Gemini 2.5 Flash Lite", description:"Lightweight version of Gemini Flash for fast inference", maxTemperature: 1.0, pricing: "free" }
    ]
  },
  mistral: {
    label: "Mistral AI",
    models: [
      { id: "mistral-large-2411", name: "Mistral Large", description: "Top-tier large model for high-complexity tasks", maxTemperature: 1.0, pricing: "premium", provider: "Mistral AI" },
      { id: "mistral-medium-2508", name: "Mistral Medium", description: "Medium model with excellent performance", maxTemperature: 1.0, pricing: "standard", provider: "Mistral AI" },
      { id: "magistral-medium-2507", name: "Magistral Medium", description: "Medium sized reasoning model", maxTemperature: 1.0, pricing: "standard", provider: "Mistral AI" },
      { id: "mistral-small-latest", name: "Mistral Small", description: "Updated small model with excellent performance", maxTemperature: 1.0, pricing: "standard", provider: "Mistral AI" },
      { id: "codestral-latest", name: "Codestral", description: "Specialized model for coding tasks", maxTemperature: 1.0, pricing: "standard", provider: "Mistral AI" },
      { id: "open-mistral-nemo", name: "Mistral Nemo 12B", description: "Best multilingual open source model", maxTemperature: 1.0, pricing: "free", provider: "Mistral AI" },
      { id: "ministral-8b-latest", name: "Ministral 8B", description: "Powerful edge model with high performance/price ratio", maxTemperature: 1.0, pricing: "free", provider: "Mistral AI" }
    ]
  },
  openrouter: {
    label: "OpenRouter Models",
    models: [
      // Free models
      { id: "mistralai/mistral-7b-instruct:free", name: "Mistral 7B Instruct", description: "Efficient instruction-following model", maxTemperature: 1.0, pricing: "free", provider: "Mistral AI" },
      { id: "mistralai/mistral-small-3.2-24b-instruct:free", name: "Mistral small 3.2 24B Instruct", description: "Efficient instruction-following model", maxTemperature: 1.0, pricing: "free", provider: "Mistral AI" },
      { id: "z-ai/glm-4.5-air:free", name: "GLM 4.5 Air", description: "GLM 4.5 Air from Z AI", maxTemperature: 2.0, pricing: "free", provider: "Z AI" },
      { id: "moonshotai/kimi-k2:free", name: "Kimi K2", description: "Massive 1T mixture of experts model", maxTemperature: 2.0, pricing: "free", provider: "Moonshot AI" },
      { id: "qwen/qwen3-235b-a22b:free", name: "Qwen3 235B A22B", description: "Qwen 3 MoE with 235B parameters", maxTemperature: 2.0, pricing: "free", provider: "Qwen" },
      { id: "qwen/qwen3-4b:free", name: "Qwen3 4B", description: "Effcient implementation of Qwen 3 with 4B parameters", maxTemperature: 2.0, pricing: "free", provider: "Qwen" },
      { id: "deepseek/deepseek-chat-v3-0324:free", name: "Deepseek V3 0324", description: "Deepseek V3 (non-reasoning)", maxTemperature: 1.0, pricing: "free", provider: "Deepseek" },
      { id: "meta-llama/llama-3.3-70b-instruct:free", name: "Llama 3.3 70B", description: "Llama model form Meta", maxTemperature: 1.0, pricing: "free", provider: "Meta" },
  //    { id: "meta-llama/llama-3.1-8b-instruct:free", name: "Llama 3.1 8B (Free)", description: "Free Meta Llama model for general use", maxTemperature: 1.0, pricing: "free", provider: "Meta" },
      

    ]
  }
};

// Helper function to check if a model is a Gemini model
const isGeminiModel = (modelId) => {
  return modelId && modelId.startsWith('gemini');
};

// Helper function to check if a model is a Mistral model
const isMistralModel = (modelId) => {
  return modelId && (modelId.startsWith('mistral') || 
                    modelId.startsWith('open-mistral') || 
                    modelId.startsWith('codestral') ||
                    modelId.startsWith('ministral'));
};

// Helper function to get the maximum temperature for a model
const getMaxTemperatureForModel = (modelId) => {
  // Check if it's a Gemini model
  if (isGeminiModel(modelId)) {
    return 1.0;
  }
  
  // Check if it's a Mistral model
  if (isMistralModel(modelId)) {
    return 1.0;
  }
  
  // Look up the model in our categories
  for (const category of Object.values(MODEL_CATEGORIES)) {
    const model = category.models.find(m => m.id === modelId);
    if (model && model.maxTemperature !== undefined) {
      return model.maxTemperature;
    }
  }
  
  // Default for unknown models
  return 2.0;
};

// Helper function to validate and constrain temperature for a model
const constrainTemperatureForModel = (temperature, modelId) => {
  const maxTemp = getMaxTemperatureForModel(modelId);
  return Math.min(Math.max(temperature, 0.0), maxTemp);
};

const ContextProvider = (props) => {
    const [input, setInput] = useState('');
    const [recentPrompt, setRecentPrompt] = useState('');
    const [prevPrompt, setPrevPrompts] = useState(["what is React.js?"]);
    const [sessions, setSessions] = useState([]); // [{id,title,messages,createdAt}]
    const [activeSessionId, setActiveSessionId] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [loading, setLoading] = useState(false);
    const [resultData, setResultData] = useState('');
    const [messages, setMessages] = useState([]); // [{ role: 'user'|'assistant'|'model', content: string }]
    const [thinkingPhrase, setThinkingPhrase] = useState('Thinking…');
    const [thinkingPhrases, setThinkingPhrases] = useState(null);
    const [toast, setToast] = useState(null); // { message, type }
    const [theme, setTheme] = useState(() => {
        const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('theme') : null;
        return saved || 'light';
    });
    const [verbosity, setVerbosity] = useState(() => {
        try {
            const raw = typeof localStorage !== 'undefined' ? localStorage.getItem('verbosity') : null;
            const num = parseInt(raw ?? '3', 10);
            return Number.isFinite(num) ? Math.min(5, Math.max(1, num)) : 3;
        } catch { return 3; }
    });
    const [temperature, setTemperature] = useState(() => {
        try {
            const raw = typeof localStorage !== 'undefined' ? localStorage.getItem('temperature') : null;
            const savedModel = typeof localStorage !== 'undefined' ? localStorage.getItem('selectedModel') : null;
            const model = savedModel || 'gemini-2.5-flash';
            
            const num = parseFloat(raw ?? '1.0');
            const maxTemp = getMaxTemperatureForModel(model);
            const validNum = Number.isFinite(num) ? Math.min(maxTemp, Math.max(0.0, num)) : 1.0;
            
            console.log(`Initializing temperature: raw=${raw}, model=${model}, maxTemp=${maxTemp}, result=${validNum}`);
            return validNum;
        } catch { return 1.0; }
    });
    const [selectedModel, setSelectedModel] = useState(() => {
        try {
            const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('selectedModel') : null;
            return saved || 'gemini-2.5-flash';
        } catch { return 'gemini-2.5-flash'; }
    });
    
    // Add state for cancellation
    const [canCancel, setCanCancel] = useState(false);
    const [abortController, setAbortController] = useState(null);
    
    const [showAllSessions, setShowAllSessions] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [sidebarExtended, setSidebarExtended] = useState(true);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    // Create a constrained setTemperature function
    const setConstrainedTemperature = (newTemperature) => {
        const constrainedValue = constrainTemperatureForModel(newTemperature, selectedModel);
        console.log(`Setting temperature: ${newTemperature} -> ${constrainedValue} for model ${selectedModel}`);
        setTemperature(constrainedValue);
    };

    // Auth state
    const [user, setUser] = useState(null); // supabase user or null
    const [authReady, setAuthReady] = useState(false);

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        try { localStorage.setItem('theme', theme); } catch {}
    }, [theme]);

    useEffect(() => {
        try { localStorage.setItem('verbosity', String(verbosity)); } catch {}
    }, [verbosity]);

    useEffect(() => {
        try { localStorage.setItem('temperature', String(temperature)); } catch {}
    }, [temperature]);

    useEffect(() => {
        try { localStorage.setItem('selectedModel', selectedModel); } catch {}
    }, [selectedModel]);

    // Auto-constrain temperature when model changes
    useEffect(() => {
        const maxTemp = getMaxTemperatureForModel(selectedModel);
        if (temperature > maxTemp) {
            const constrainedTemp = constrainTemperatureForModel(temperature, selectedModel);
            console.log(`Constraining temperature from ${temperature} to ${constrainedTemp} for model ${selectedModel}`);
            setTemperature(constrainedTemp);
        }
    }, [selectedModel]); // Remove temperature from dependencies to avoid infinite loop

    // Prefetch thinking phrases once on mount to avoid first-click race
    useEffect(() => {
        (async () => {
            try {
                const baseUrl = import.meta.env.VITE_API_BASE_URL ||
                    (typeof window !== 'undefined' && window.location.origin !== 'null' ? window.location.origin : 'http://localhost:8000');
                console.log('Fetching thinking phrases from:', baseUrl + '/thinking'); // Debug log
                const res = await fetch(baseUrl + '/thinking');
                console.log('Fetch response status:', res.status); // Debug log
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                const data = await res.json();
                console.log('Fetched thinking phrases:', data); // Debug log
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                console.log('Processed thinking phrases array:', arr); // Debug log
                setThinkingPhrases(arr);
                // If we don't have a thinking phrase yet, set the first one
                if (!thinkingPhrase || thinkingPhrase === 'Thinking…') {
                    const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                    console.log('Setting initial thinking phrase:', pick); // Debug log
                    setThinkingPhrase(pick);
                }
            } catch (error) {
                console.error('Error fetching thinking phrases:', error); // Better error logging
                setThinkingPhrases(['Thinking…']);
                // Also set the thinking phrase to the default
                if (!thinkingPhrase || thinkingPhrase === 'Thinking…') {
                    setThinkingPhrase('Thinking…');
                }
            }
        })();
    }, []);

    // Init Supabase auth listener
    useEffect(() => {
        if (!supabase) { setAuthReady(true); return; }
        // Get initial session
        supabase.auth.getSession().then(({ data }) => {
            setUser(data?.session?.user || null);
            setAuthReady(true);
        });
        const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user || null);
            if (session?.user) {
                // On login, migrate any local session to remote
                try { migrateLocalToRemote(); } catch {}
                // Then fetch remote sessions
                try { fetchRemoteSessions(); } catch {}
            } else {
                // On logout, keep local only
                setActiveSessionId(null);
                setMessages([]);
            }
        });
        return () => { sub?.subscription?.unsubscribe?.(); };
    }, []);

    // After auth resolves, fetch sessions on first load if user exists
    useEffect(() => {
        if (authReady && user) {
            fetchRemoteSessions();
        }
    }, [authReady, user]);

    // Do not create any session on first load; show landing until user selects or sends

    
    const setSelectedTheme = (newTheme) => setTheme(newTheme);
    
    // Maintain backward compatibility
    const darkMode = theme === 'dark';
    const toggleDarkMode = () => setTheme(theme === 'dark' ? 'light' : 'dark');
    
    const openSettings = () => {
        setIsSettingsOpen(true);
    };
    
    const closeSettings = () => {
        setIsSettingsOpen(false);
    };

    const showToast = (message, type = 'info', ms = 2000) => {
        setToast({ message, type });
        if (showToast._t) clearTimeout(showToast._t);
        showToast._t = setTimeout(() => setToast(null), ms);
    };

    const handleApiResponse = (response, sid, isEdit = false, editedHistory = null, index) => {
        const responseHtml = marked.parse(response?.text || response || '', { 
            breaks: true, 
            gfm: true, 
            headerIds: false, 
            mangle: false,
            pedantic: false,
            smartLists: true
        });
        const assistantTurn = { role: 'assistant', content: responseHtml, artifacts: response?.artifacts || [] };

        if (isEdit) {
            const finalMessages = [...editedHistory, assistantTurn];
            setMessages(finalMessages);
            setSessions(prev => prev.map(x => x.id === sid ? { ...x, messages: finalMessages } : x));
            if (user && sid && index !== undefined) {
                try { persistMessage(sid, assistantTurn, index); } catch {}
            }
        } else {
            setMessages(prev => {
                const out = [...prev];
                const idx = out.length - 1;
                if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                return out;
            });
            setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
            if (user && sid && index !== undefined) {
                try { persistMessage(sid, assistantTurn, index); } catch {}
            }
        }
    };

    const newChat = async () => {
        // Reset UI to landing/new chat. Do not create DB rows.
        setActiveSessionId(null);
        setLoading(false);
        setShowResult(false);
        setRecentPrompt('');
        setResultData('');
        setMessages([]);
    }
    

    const addUploadedFile = (file) => {
        setUploadedFiles(prev => [...prev, file]);
    };

    const removeUploadedFile = (fileName) => {
        setUploadedFiles(prev => prev.filter(f => f.name !== fileName));
    };

    const onSent = async (prompt, file = null) => {
        // Determine prompt text
        const text = (prompt ?? input ?? '').trim();
        if (!text && !file) return;

        if (file) {
            addUploadedFile(file);
        }

        // Clear input and show loading
        setInput('');
        setResultData('')
        setLoading(true)
        setShowResult(true)
        
        // Pick a thinking phrase lazily
        // Pick a thinking phrase
        try {
            console.log('Current thinkingPhrases state:', thinkingPhrases); // Debug log
            if (!thinkingPhrases || thinkingPhrases.length === 0) {
                // If we don't have phrases yet, fetch them
                console.log('Fetching thinking phrases...'); // Debug log
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

                const baseUrl = getBackendUrl();
                console.log('Fetching from:', baseUrl + '/thinking'); // Debug log
                const res = await fetch(baseUrl + '/thinking');
                console.log('Fetch response status:', res.status); // Debug log
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                const data = await res.json();
                console.log('Fetched thinking phrases data:', data); // Debug log
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                console.log('Processed phrases array:', arr); // Debug log
                setThinkingPhrases(arr);
                const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                console.log('Selected phrase (from fetch):', pick); // Debug log
                setThinkingPhrase(pick);
            } else {
                // Use existing phrases
                const pick = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)] || 'Thinking…';
                console.log('Selected phrase (from state):', pick); // Debug log
                setThinkingPhrase(pick);
            }
        } catch (error) {
            console.error('Error picking thinking phrase:', error);
            setThinkingPhrase('Thinking…');
        }
        
        // Enable cancellation when thinking phrase appears
        setCanCancel(true);
        
        // Create AbortController for this request
        const controller = new AbortController();
        setAbortController(controller);

        setRecentPrompt(text)

        const userTurn = { role: 'user', content: text };
        const nextMessages = [...messages, userTurn];
        const historyForApi = nextMessages.map(m => ({
            role: m.role,
            content: m.role === 'assistant' ? (m.content.replace(/<[^>]*>?/gm, '')) : m.content,
        }));

        setMessages(prev => [...prev, userTurn, { role: 'assistant', content: '' }]);

        let sid = activeSessionId;
        if (!sid) {
            sid = (globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2));
            setActiveSessionId(sid);
            const newSession = {
                id: sid,
                title: text.slice(0, 60) || 'New chat',
                messages: [...nextMessages, { role: 'assistant', content: '' }],
                createdAt: Date.now()
            };
            setSessions(prev => [newSession, ...prev]);

            if (user) {
                try {
                    await supabase.from('chat_sessions').insert({ id: sid, user_id: user.id, title: newSession.title });
                } catch {}
            }
        } else {
            setSessions(prev => prev.map(s => s.id === sid ? {
                ...s,
                messages: [...nextMessages, { role: 'assistant', content: '' }],
                title: (!s.manualTitle && (!s.title || s.title === 'New chat')) ? (text.slice(0, 60) || s.title) : s.title,
            } : s));
        }

        if (user && sid) {
            try {
                await persistMessage(sid, userTurn, messages.length);
                await supabase.from('chat_sessions').update({ updated_at: new Date().toISOString() }).eq('id', sid).eq('user_id', user.id);
                const live = sessions.find(x => x.id === sid) || {};
                if (!live.manualTitle && (!live.title || live.title === 'New chat')) {
                    await persistSessionTitleIfNeeded(sid, text);
                }
            } catch {}
        }

        const assistantMessageIndex = messages.length + 1;
        let assistantMessage = '';
        let assistantArtifacts = [];
        
        // Initialize markdown buffer manager for safe streaming
        const markdownBuffer = new MarkdownBufferManager();
        
        // onDelta callback for streaming events
        const onDelta = (data) => {
            try {
                switch (data.type) {
                    case 'delta':
                        // Use markdown buffer manager for safe parsing
                        const renderInfo = markdownBuffer.addToken(data.text);
                        assistantMessage = renderInfo.rawContent;
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: renderInfo.rawContent, // Store raw content instead
                                    pendingText: '',
                                    isStreaming: true,
                                    isComplete: renderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'status':
                        // Update thinking phrase with tool status
                        setThinkingPhrase(data.message);
                        break;
                    case 'artifacts':
                        // Add artifacts to current message
                        assistantArtifacts = [...assistantArtifacts, ...data.artifacts];
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                const currentRenderInfo = markdownBuffer.getSafeContent();
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: currentRenderInfo.content, 
                                    pendingText: currentRenderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: currentRenderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'error':
                        console.error('Streaming error:', data.message);
                        setLoading(false);
                        break;
                    case 'done':
                        // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                        const assistantTurn = { role: 'assistant', content: assistantMessage, artifacts: assistantArtifacts };
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                            return out;
                        });
                        
                        setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                        
                        if (user && sid && assistantMessageIndex !== undefined) {
                            try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                        }
                        
                        setLoading(false);
                        break;
                }
            } catch (error) {
                console.error('Error processing streaming event:', error);
                // Graceful degradation - fallback to non-streaming
                fallbackToNonStreaming();
            }
        };
        
        const fallbackToNonStreaming = async () => {
            try {
                console.log('Falling back to non-streaming API');
                const response = await runChatWithHistory(historyForApi, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, file);
                
                if (response.text) {
                    // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                    const assistantTurn = { role: 'assistant', content: response.text, artifacts: response.artifacts || [] };
                    
                    setMessages(prev => {
                        const out = [...prev];
                        const idx = out.length - 1;
                        if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                        return out;
                    });
                    
                    setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                    
                    if (user && sid && assistantMessageIndex !== undefined) {
                        try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                    }
                }
            } catch (error) {
                console.error('Error in fallback non-streaming:', error);
                const fallback = "Sorry, I can't complete that request. Please try again.";
                handleApiResponse(fallback, sid, false, null, assistantMessageIndex);
            } finally {
                setLoading(false);
            }
        };
        
        try {
            // Use streaming endpoint with abort signal
            await streamChatWithHistory(historyForApi, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, file, onDelta, controller.signal);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request was cancelled');
                // Handle cancellation - don't set canCancel to false here as it's handled in finally
                setLoading(false);
                return;
            }
            console.error('Error in onSent (streaming):', error);
            // Fallback to non-streaming on error
            await fallbackToNonStreaming();
        } finally {
            setCanCancel(false);
            setAbortController(null);
        }
    }

    // Add cancellation function
    const cancelCurrentRequest = () => {
        if (abortController) {
            abortController.abort();
        }
        // Set canCancel to false immediately for UI update
        setCanCancel(false);
    };

    // Switch active session and sync UI state/messages
    const setActiveSession = async (id) => {
        setActiveSessionId(id);
        setResultData('');
        setLoading(true);
        try {
            // Always fetch fresh messages from DB for reliability
            if (supabase && user) {
                const { data: msgs } = await supabase
                    .from('chat_messages')
                    .select('role, content, idx, created_at')
                    .eq('session_id', id)
                    .order('idx', { ascending: true });
                const arr = (msgs || []).map(m => ({ role: m.role, content: m.content }));
                setMessages(arr);
                setSessions(prev => prev.map(s => s.id === id ? ({ ...s, messages: arr }) : s));
                setShowResult(arr.length > 0);
                const lastUser = [...arr].reverse().find(m => m.role === 'user');
                setRecentPrompt(lastUser?.content || '');
            }
        } finally {
            setLoading(false);
        }
    }

    // Supabase helpers: persistence and auth
    const fetchRemoteSessions = async () => {
        if (!supabase || !user) return;
        const { data: sessionsData } = await supabase
            .from('chat_sessions')
            .select('id, title, pinned, created_at')
            .eq('user_id', user.id)
            .order('pinned', { ascending: false })
            .order('created_at', { ascending: false });
        const out = [];
        for (const s of (sessionsData || [])) {
            const { data: msgs } = await supabase
                .from('chat_messages')
                .select('role, content, idx, created_at')
                .eq('session_id', s.id)
                .order('idx', { ascending: true });
            out.push({ id: s.id, title: s.title || 'New chat', manualTitle: (s.title && s.title !== 'New chat') || false, pinned: !!s.pinned, createdAt: new Date(s.created_at).getTime(), messages: (msgs || []).map(m => ({ role: m.role, content: m.content })) });
        }
        // Populate sidebar; do not auto-select a session. Keep landing empty.
        setSessions(out);
    };

    const persistSessionStub = async (id, title) => {
        if (!supabase || !user) return;
        await supabase.from('chat_sessions').upsert({ id, user_id: user.id, title: title || 'New chat' }, { onConflict: 'id' });
    };
    const persistSessionTitleIfNeeded = async (id, firstPrompt) => {
        if (!supabase || !user || !id) return;
        const title = String(firstPrompt || '').slice(0, 60) || 'New chat';
        await supabase.from('chat_sessions').update({ title }).eq('id', id).eq('user_id', user.id);
    };
    const persistMessage = async (sessionId, msg, idx) => {
        if (!supabase || !user || !sessionId) return;
        await supabase.from('chat_messages').insert({ session_id: sessionId, role: msg.role, content: msg.content, idx: idx ?? 0 });
    };

    const migrateLocalToRemote = async () => {
        try {
            if (!user || !sessions.length) return;
            // Only migrate non-UUID-like temporary ids (but ours are UUID-ish). We'll always upsert stub and check if remote msgs exist.
            const current = sessions.find(s => s.id === activeSessionId) || sessions[0];
            if (!current) return;
            await persistSessionStub(current.id, current.title);
            // Check if messages already exist
            const { data: existing } = await supabase
                .from('chat_messages')
                .select('id')
                .eq('session_id', current.id)
                .limit(1);
            if (existing && existing.length > 0) return; // already migrated
            for (let i = 0; i < (current.messages || []).length; i++) {
                const m = current.messages[i];
                await persistMessage(current.id, m, i);
            }
        } catch (e) { /* ignore */ }
    };

    const signInWithGoogle = async () => {
        if (!supabase) return;
        await supabase.auth.signInWithOAuth({ provider: 'google' });
    };
    const signOut = async () => {
        if (!supabase) return;
        await supabase.auth.signOut();
    };

    const contextValue = {
        prevPrompt,
        setPrevPrompts,
        onSent,
        setRecentPrompt,
        recentPrompt,
        showResult,
        loading,
        resultData,
        thinkingPhrase,
        input,
        setInput,
        newChat,
        messages,
        setMessages,
        sessions,
        setSessions,
        activeSessionId,
        setActiveSessionId,
        setActiveSession,
        editUserMessageAndRegenerate: undefined, // placeholder, will be set below
        // auth
        user,
        authReady,
        signInWithGoogle,
        signOut,
        redoAssistantAt: undefined,
        copyAssistantAt: undefined,
        shareAssistantAt: undefined,
        verifyAssistantAt: undefined,
        toast,
        showToast,
        theme,
        setSelectedTheme,
        darkMode,
        toggleDarkMode,
        verbosity,
        setVerbosity,
        temperature,
        setTemperature: setConstrainedTemperature,
        showAllSessions,
        setShowAllSessions,
        uploadedFiles,
        addUploadedFile,
        removeUploadedFile,
        sidebarExtended,
        setSidebarExtended,
        isSettingsOpen,
        openSettings,
        closeSettings,
        selectedModel,
        setSelectedModel,
        modelCategories: MODEL_CATEGORIES,
        // Helper functions for temperature constraints
        isGeminiModel,
        isMistralModel,
        getMaxTemperatureForModel,
        constrainTemperatureForModel,
        canCancel,
        cancelCurrentRequest
    }

    // Add editing function after we have access to state setters
    contextValue.editUserMessageAndRegenerate = async (userIndex, newContent) => {
        const sid = activeSessionId || (sessions[0]?.id);
        if (!sid) return;
        const s = sessions.find(x => x.id === sid);
        if (!s) return;
        if (userIndex < 0 || userIndex >= s.messages.length) return;
        if (s.messages[userIndex]?.role !== 'user') return;

        const edited = s.messages.slice(0, userIndex + 1).map((m, i) =>
            i === userIndex ? { ...m, content: newContent.trim() } : m
        );

        const newTitle = userIndex === 0
            ? (newContent.trim().slice(0, 60) || 'New chat')
            : s.title;

        const updatedSession = { ...s, messages: edited, title: newTitle };
        setSessions(prev => prev.map(x => x.id === sid ? updatedSession : x));
        setShowResult(true);
        setRecentPrompt(newContent.trim());
        setResultData('');

        // Append assistant placeholder atomically to ensure order
        const nextMessages = [...edited, { role: 'assistant', content: '' }];
        setMessages(nextMessages);
        if (sid) setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: nextMessages }) : s));

        setLoading(true);

        // Pick a thinking phrase lazily
        // Pick a thinking phrase
        try {
            if (!thinkingPhrases || thinkingPhrases.length === 0) {
                // If we don't have phrases yet, fetch them
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

                const baseUrl = getBackendUrl();
                const res = await fetch(baseUrl + '/thinking');
                const data = await res.json();
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                setThinkingPhrases(arr);
                const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            } else {
                // Use existing phrases
                const pick = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            }
        } catch (error) {
            console.error('Error picking thinking phrase:', error);
            setThinkingPhrase('Thinking…');
        }

        const assistantMessageIndex = edited.length;
        let assistantMessage = '';
        let assistantArtifacts = [];
        
        // Initialize markdown buffer manager for safe streaming
        const markdownBuffer = new MarkdownBufferManager();
        
        // onDelta callback for streaming events
        const onDelta = (data) => {
            try {
                switch (data.type) {
                    case 'delta':
                        const renderInfo = markdownBuffer.addToken(data.text);
                        assistantMessage = renderInfo.rawContent;
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: renderInfo.content, 
                                    pendingText: renderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: renderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'status':
                        setThinkingPhrase(data.message);
                        break;
                    case 'artifacts':
                        assistantArtifacts = [...assistantArtifacts, ...data.artifacts];
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                const currentRenderInfo = markdownBuffer.getSafeContent();
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: currentRenderInfo.content, 
                                    pendingText: currentRenderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: currentRenderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'error':
                        console.error('Streaming error:', data.message);
                        setLoading(false);
                        break;
                    case 'done':
                        // Store raw markdown content instead of pre-parsed HTML to preserve formatting  
                        const assistantTurn = { role: 'assistant', content: assistantMessage, artifacts: assistantArtifacts };
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                            return out;
                        });
                        
                        setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                        
                        if (user && sid && assistantMessageIndex !== undefined) {
                            try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                        }
                        
                        setLoading(false);
                        break;
                }
            } catch (error) {
                console.error('Error processing streaming event:', error);
                fallbackToNonStreaming();
            }
        };
        
        const fallbackToNonStreaming = async () => {
            try {
                console.log('Falling back to non-streaming API for edit');
                const response = await runChatWithHistory(edited, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null);
                
                if (response.text) {
                    // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                    const assistantTurn = { role: 'assistant', content: response.text, artifacts: response.artifacts || [] };
                    
                    setMessages(prev => {
                        const out = [...prev];
                        const idx = out.length - 1;
                        if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                        return out;
                    });
                    
                    setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                    
                    if (user && sid && assistantMessageIndex !== undefined) {
                        try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                    }
                }
            } catch (error) {
                console.error('Error in fallback non-streaming for edit:', error);
                const fallback = "Sorry, I can't complete that request. Please try again.";
                handleApiResponse(fallback, sid, true, edited, assistantMessageIndex);
            } finally {
                setLoading(false);
            }
        };
        
        try {
            await streamChatWithHistory(edited, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null, onDelta);
        } catch (error) {
            console.error('Error in editUserMessageAndRegenerate (streaming):', error);
            await fallbackToNonStreaming();
        }
    };

    // Actions under assistant messages
    contextValue.copyAssistantAt = async (index) => {
        const msg = messages[index];
        if (!msg || msg.role !== 'assistant') return;
        const tmp = document.createElement('div');
        tmp.innerHTML = msg.content || '';
        const text = tmp.textContent || tmp.innerText || '';
        try { await navigator.clipboard.writeText(text); showToast('Copied to clipboard', 'success'); } catch { showToast('Copy failed', 'error'); }
    };

    contextValue.shareAssistantAt = async (index) => {
        const sid = activeSessionId || (sessions[0]?.id);
        const msg = messages[index];
        if (!msg || msg.role !== 'assistant') return;
        const payload = {
            sessionId: sid,
            index,
            role: msg.role,
            content: msg.content,
            createdAt: Date.now(),
        };
        const text = JSON.stringify(payload, null, 2);
        if (navigator.share) {
            try { await navigator.share({ title: 'ESI response', text }); showToast('Shared', 'success'); return; } catch {}
        }
        try { await navigator.clipboard.writeText(text); showToast('Link copied', 'success'); } catch { showToast('Share failed', 'error'); }
    };

    contextValue.redoAssistantAt = async (index) => {
        // Find nearest prior user message
        let cut = -1;
        for (let i = index - 1; i >= 0; i--) {
            if (messages[i]?.role === 'user') { cut = i; break; }
        }
        if (cut < 0) return;
        const truncated = messages.slice(0, cut + 1);
        // Prepare assistant placeholder
        setMessages([...truncated, { role: 'assistant', content: '' }]);
        // Persist to session
        const sid = activeSessionId || (sessions[0]?.id);
        if (sid) setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...truncated, { role: 'assistant', content: '' }] }) : s));

        setLoading(true);
        setResultData('');
        showToast('Regenerating…', 'info');
        // Pick a thinking phrase lazily
        // Pick a thinking phrase
        try {
            if (!thinkingPhrases || thinkingPhrases.length === 0) {
                // If we don't have phrases yet, fetch them
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

                const baseUrl = getBackendUrl();
                const res = await fetch(baseUrl + '/thinking');
                const data = await res.json();
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                setThinkingPhrases(arr);
                const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            } else {
                // Use existing phrases
                const pick = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            }
        } catch (error) {
            console.error('Error picking thinking phrase:', error);
            setThinkingPhrase('Thinking…');
        }

        // Enable cancellation when thinking phrase appears
        setCanCancel(true);
        
        // Create AbortController for this request
        const controller = new AbortController();
        setAbortController(controller);

        // Clean history (strip HTML from prior assistant messages)
        const stripHtml = (html) => {
            if (typeof html !== 'string') return '';
            const tmp = document.createElement('div');
            tmp.innerHTML = html;
            return (tmp.textContent || tmp.innerText || '').trim();
        };
        const cleanHistory = truncated.map(m => ({
            role: m.role,
            content: m.role === 'assistant' ? stripHtml(m.content) : (m.content || ''),
        }));

        const assistantMessageIndex = truncated.length;
        let assistantMessage = '';
        let assistantArtifacts = [];
        
        // Initialize markdown buffer manager for safe streaming
        const markdownBuffer = new MarkdownBufferManager();
        
        // onDelta callback for streaming events
        const onDelta = (data) => {
            try {
                switch (data.type) {
                    case 'delta':
                        const renderInfo = markdownBuffer.addToken(data.text);
                        assistantMessage = renderInfo.rawContent;
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: renderInfo.content, 
                                    pendingText: renderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: renderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'status':
                        setThinkingPhrase(data.message);
                        break;
                    case 'artifacts':
                        assistantArtifacts = [...assistantArtifacts, ...data.artifacts];
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                const currentRenderInfo = markdownBuffer.getSafeContent();
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: currentRenderInfo.content, 
                                    pendingText: currentRenderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: currentRenderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'error':
                        console.error('Streaming error:', data.message);
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                    case 'done':
                        // Store raw markdown content instead of pre-parsed HTML to preserve formatting  
                        const assistantTurn = { role: 'assistant', content: assistantMessage, artifacts: assistantArtifacts };
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                            return out;
                        });
                        
                        setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                        
                        if (user && sid && assistantMessageIndex !== undefined) {
                            try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                        }
                        
                        showToast('Regenerated', 'success');
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                }
            } catch (error) {
                console.error('Error processing streaming event:', error);
                fallbackToNonStreaming();
            }
        };
        
        const fallbackToNonStreaming = async () => {
            try {
                console.log('Falling back to non-streaming API for redo');
                const response = await runChatWithHistory(cleanHistory, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null);
                
                if (response.text) {
                    // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                    const assistantTurn = { role: 'assistant', content: response.text, artifacts: response.artifacts || [] };
                    
                    setMessages(prev => {
                        const out = [...prev];
                        const idx = out.length - 1;
                        if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                        return out;
                    });
                    
                    setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                    
                    if (user && sid && assistantMessageIndex !== undefined) {
                        try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                    }
                }
                showToast('Regenerated', 'success');
            } catch (error) {
                console.error('Error in fallback non-streaming for redo:', error);
                const fallback = "Sorry, I can't complete that request. Please try again.";
                handleApiResponse(fallback, sid, false, null, assistantMessageIndex);
            } finally {
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
            }
        };
        
        try {
            await streamChatWithHistory(cleanHistory, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null, onDelta, controller.signal);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request was cancelled');
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
                return;
            }
            console.error('Error in redoAssistantAt (streaming):', error);
            await fallbackToNonStreaming();
        }

    };

    contextValue.verifyAssistantAt = async (index) => {
        const upto = messages.slice(0, index + 1);
        const prompt = 'Double-check the previous assistant response. Identify any factual errors or missing citations. Provide corrected information with sources.';
        const verifyHistory = [...upto, { role: 'user', content: prompt }];
        // Insert assistant placeholder so thinking phrase can show in bubble
        const sid = activeSessionId || (sessions[0]?.id);
        const withPlaceholder = [...verifyHistory, { role: 'assistant', content: '' }];
        setMessages(withPlaceholder);
        if (sid) setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: withPlaceholder }) : s));

        setLoading(true);
        setResultData('');
        showToast('Verifying…', 'info');
        // Pick a thinking phrase lazily
        // Pick a thinking phrase
        try {
            if (!thinkingPhrases || thinkingPhrases.length === 0) {
                // If we don't have phrases yet, fetch them
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

                const baseUrl = getBackendUrl();
                const res = await fetch(baseUrl + '/thinking');
                const data = await res.json();
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                setThinkingPhrases(arr);
                const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            } else {
                // Use existing phrases
                const pick = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            }
        } catch (error) {
            console.error('Error picking thinking phrase:', error);
            setThinkingPhrase('Thinking…');
        }
        
        // Enable cancellation when thinking phrase appears
        setCanCancel(true);
        
        // Create AbortController for this request
        const controller = new AbortController();
        setAbortController(controller);

        let assistantMessage = '';
        let assistantArtifacts = [];
        
        // Initialize markdown buffer manager for safe streaming
        const markdownBuffer = new MarkdownBufferManager();
        
        // onDelta callback for streaming events
        const onDelta = (data) => {
            try {
                switch (data.type) {
                    case 'delta':
                        const renderInfo = markdownBuffer.addToken(data.text);
                        assistantMessage = renderInfo.rawContent;
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: renderInfo.content, 
                                    pendingText: renderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: renderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'status':
                        setThinkingPhrase(data.message);
                        break;
                    case 'artifacts':
                        assistantArtifacts = [...assistantArtifacts, ...data.artifacts];
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                const currentRenderInfo = markdownBuffer.getSafeContent();
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: currentRenderInfo.content, 
                                    pendingText: currentRenderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: currentRenderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'error':
                        console.error('Streaming error:', data.message);
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                    case 'done':
                        // Store raw markdown content instead of pre-parsed HTML to preserve formatting  
                        const assistantTurn = { role: 'assistant', content: assistantMessage, artifacts: assistantArtifacts };
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                            return out;
                        });
                        
                        setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                        
                        if (user && sid) {
                            try { persistMessage(sid, assistantTurn, messages.length); } catch {}
                        }
                        
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                }
            } catch (error) {
                console.error('Error processing streaming event:', error);
                fallbackToNonStreaming();
            }
        };
        
        const fallbackToNonStreaming = async () => {
            try {
                console.log('Falling back to non-streaming API for verify');
                const response = await runChatWithHistory(verifyHistory, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null);
                
                if (response.text) {
                    // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                    const assistantTurn = { role: 'assistant', content: response.text, artifacts: response.artifacts || [] };
                    
                    setMessages(prev => {
                        const out = [...prev];
                        const idx = out.length - 1;
                        if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                        return out;
                    });
                    
                    setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                    
                    if (user && sid) {
                        try { persistMessage(sid, assistantTurn, messages.length); } catch {}
                    }
                }
                showToast('Verification added', 'success');
            } catch (error) {
                console.error('Error in fallback non-streaming for verify:', error);
                const fallback = "Sorry, I can't complete that request. Please try again.";
                handleApiResponse(fallback, sid, false, null, messages.length);
            } finally {
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
            }
        };
        
        try {
            await streamChatWithHistory(verifyHistory, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null, onDelta, controller.signal);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request was cancelled');
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
                return;
            }
            console.error('Error in verifyAssistantAt (streaming):', error);
            await fallbackToNonStreaming();
        }
    };

    // Add editing function after we have access to state setters
    contextValue.editUserMessageAndRegenerate = async (userIndex, newContent) => {
        const sid = activeSessionId || (sessions[0]?.id);
        if (!sid) return;
        const s = sessions.find(x => x.id === sid);
        if (!s) return;
        if (userIndex < 0 || userIndex >= s.messages.length) return;
        if (s.messages[userIndex]?.role !== 'user') return;

        const edited = s.messages.slice(0, userIndex + 1).map((m, i) =>
            i === userIndex ? { ...m, content: newContent.trim() } : m
        );

        const newTitle = userIndex === 0
            ? (newContent.trim().slice(0, 60) || 'New chat')
            : s.title;

        const updatedSession = { ...s, messages: edited, title: newTitle };
        setSessions(prev => prev.map(x => x.id === sid ? updatedSession : x));
        setShowResult(true);
        setRecentPrompt(newContent.trim());
        setResultData('');

        // Append assistant placeholder atomically to ensure order
        const nextMessages = [...edited, { role: 'assistant', content: '' }];
        setMessages(nextMessages);
        if (sid) setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: nextMessages }) : s));

        setLoading(true);

        // Pick a thinking phrase lazily
        // Pick a thinking phrase
        try {
            if (!thinkingPhrases || thinkingPhrases.length === 0) {
                // If we don't have phrases yet, fetch them
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

                const baseUrl = getBackendUrl();
                const res = await fetch(baseUrl + '/thinking');
                const data = await res.json();
                const arr = Array.isArray(data?.phrases) ? data.phrases : ['Thinking…'];
                setThinkingPhrases(arr);
                const pick = arr[Math.floor(Math.random() * arr.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            } else {
                // Use existing phrases
                const pick = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)] || 'Thinking…';
                setThinkingPhrase(pick);
            }
        } catch (error) {
            console.error('Error picking thinking phrase:', error);
            setThinkingPhrase('Thinking…');
        }
        
        // Enable cancellation when thinking phrase appears
        setCanCancel(true);
        
        // Create AbortController for this request
        const controller = new AbortController();
        setAbortController(controller);

        const assistantMessageIndex = edited.length;
        let assistantMessage = '';
        let assistantArtifacts = [];
        
        // Initialize markdown buffer manager for safe streaming
        const markdownBuffer = new MarkdownBufferManager();
        
        // onDelta callback for streaming events
        const onDelta = (data) => {
            try {
                switch (data.type) {
                    case 'delta':
                        const renderInfo = markdownBuffer.addToken(data.text);
                        assistantMessage = renderInfo.rawContent;
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: renderInfo.content, 
                                    pendingText: renderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: renderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'status':
                        setThinkingPhrase(data.message);
                        break;
                    case 'artifacts':
                        assistantArtifacts = [...assistantArtifacts, ...data.artifacts];
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') {
                                const currentRenderInfo = markdownBuffer.getSafeContent();
                                out[idx] = { 
                                    role: 'assistant', 
                                    content: currentRenderInfo.content, 
                                    pendingText: currentRenderInfo.pendingText,
                                    isStreaming: true,
                                    isComplete: currentRenderInfo.isComplete,
                                    artifacts: assistantArtifacts 
                                };
                            }
                            return out;
                        });
                        break;
                    case 'error':
                        console.error('Streaming error:', data.message);
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                    case 'done':
                        // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                        const assistantTurn = { role: 'assistant', content: assistantMessage, artifacts: assistantArtifacts };
                        
                        setMessages(prev => {
                            const out = [...prev];
                            const idx = out.length - 1;
                            if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                            return out;
                        });
                        
                        setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                        
                        if (user && sid && assistantMessageIndex !== undefined) {
                            try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                        }
                        
                        setLoading(false);
                        setCanCancel(false);
                        setAbortController(null);
                        break;
                }
            } catch (error) {
                console.error('Error processing streaming event:', error);
                fallbackToNonStreaming();
            }
        };
        
        const fallbackToNonStreaming = async () => {
            try {
                console.log('Falling back to non-streaming API for edit');
                const response = await runChatWithHistory(edited, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null);
                
                if (response.text) {
                    // Store raw markdown content instead of pre-parsed HTML to preserve formatting
                    const assistantTurn = { role: 'assistant', content: response.text, artifacts: response.artifacts || [] };
                    
                    setMessages(prev => {
                        const out = [...prev];
                        const idx = out.length - 1;
                        if (idx >= 0 && out[idx].role === 'assistant') out[idx] = assistantTurn;
                        return out;
                    });
                    
                    setSessions(prev => prev.map(s => s.id === sid ? ({ ...s, messages: [...s.messages.slice(0, -1), assistantTurn] }) : s));
                    
                    if (user && sid && assistantMessageIndex !== undefined) {
                        try { persistMessage(sid, assistantTurn, assistantMessageIndex); } catch {}
                    }
                }
            } catch (error) {
                console.error('Error in fallback non-streaming for edit:', error);
                const fallback = "Sorry, I can't complete that request. Please try again.";
                handleApiResponse(fallback, sid, true, edited, assistantMessageIndex);
            } finally {
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
            }
        };
        
        try {
            await streamChatWithHistory(edited, { verbosity, temperature: constrainTemperatureForModel(temperature, selectedModel), model: selectedModel }, null, onDelta, controller.signal);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request was cancelled');
                setLoading(false);
                setCanCancel(false);
                setAbortController(null);
                return;
            }
            console.error('Error in editUserMessageAndRegenerate (streaming):', error);
            await fallbackToNonStreaming();
        }
    };

    // Session management utilities for sidebar menu
    contextValue.pinSession = (id) => {
        setSessions(prev => prev.map(s => s.id === id ? ({ ...s, pinned: !s.pinned }) : s)
            .sort((a,b) => (b.pinned?1:0) - (a.pinned?1:0) || (b.createdAt - a.createdAt))
        );
        showToast('Pin toggled', 'success');
    };
    contextValue.renameSession = async (id) => {
        const s = sessions.find(x => x.id === id);
        const current = s?.title || '';
        const next = prompt('Rename chat:', current);
        if (next == null) return;
        contextValue.renameSessionWithValue(id, next);
    };
    contextValue.renameSessionWithValue = async (id, value) => {
        const title = String(value ?? '').trim() || 'New chat';
        setSessions(prev => prev.map(x => x.id === id ? ({ ...x, title, manualTitle: title !== 'New chat' }) : x));
        try {
            if (supabase && user) {
                await supabase.from('chat_sessions').update({ title }).eq('id', id).eq('user_id', user.id);
            }
        } catch {}
        showToast('Renamed', 'success');
    };
    contextValue.deleteSession = async (id) => {
        try {
            if (supabase && user) {
                await supabase
                    .from('chat_sessions')
                    .delete()
                    .eq('id', id)
                    .eq('user_id', user.id);
                // chat_messages will cascade via FK on delete
            }
        } catch {}
        setSessions(prev => prev.filter(s => s.id !== id));
        if (activeSessionId === id) {
            const next = sessions.find(x => x.id !== id);
            setActiveSessionId(next?.id || null);
            setMessages(next?.messages || []);
        }
        showToast('Deleted', 'success');
    };

    // Expose actions globally for the simple handler wiring
    if (typeof window !== 'undefined') {
        window.appCtx = {
            redoAssistantAt: contextValue.redoAssistantAt,
            copyAssistantAt: contextValue.copyAssistantAt,
            shareAssistantAt: contextValue.shareAssistantAt,
            verifyAssistantAt: contextValue.verifyAssistantAt,
            pinSession: contextValue.pinSession,
            renameSession: contextValue.renameSession,
            renameSessionWithValue: contextValue.renameSessionWithValue,
            deleteSession: contextValue.deleteSession,
        };
    }

    return (
        <Context.Provider value={contextValue}>
            {props.children}
        </Context.Provider>
    )

}

export default ContextProvider