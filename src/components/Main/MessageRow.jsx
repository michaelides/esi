import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import StreamingMarkdownRenderer from '../StreamingMarkdownRenderer.jsx';
import { marked } from 'marked';

// Helper function to determine if content is already HTML or raw markdown
const isContentHTML = (content) => {
  if (!content) return false;
  
  // Check for common HTML tags that marked.parse() generates
  const htmlTags = /<(p|h[1-6]|ul|ol|li|code|pre|strong|em|a|blockquote|table|tr|td|th|div|span|br)\b[^>]*>/i;
  
  // If it contains HTML tags, do more careful detection
  if (htmlTags.test(content)) {
    // Count HTML tags vs markdown indicators
    const htmlTagCount = (content.match(htmlTags) || []).length;
    const markdownIndicators = /^(#{1,6}\s|\*\*|\*|```|\[.*\]\(|>\s|\d+\.\s|[-*+]\s)/gm;
    const markdownCount = (content.match(markdownIndicators) || []).length;
    
    // If there are more HTML tags than markdown indicators, likely HTML
    return htmlTagCount > markdownCount;
  }
  
  return false;
};

// Helper function to render markdown content safely
const renderMarkdownContent = (content, isStreaming = false) => {
  if (!content) return '';
  
  // If it's already HTML, return as-is
  if (isContentHTML(content)) {
    return content;
  }
  
  // For streaming content, just preserve line breaks without full markdown parsing
  // This prevents corrupting code blocks and other formatted content during streaming
  if (isStreaming) {
    // Preprocess the text to ensure proper handling of list items
    // Add line breaks before list markers that follow sentences
    const processedContent = content
      .replace(/\. \* /g, '.\n\n* ')  // Add line breaks before list items
      .replace(/\. \*\*/g, '.\n\n**'); // Handle bold list items as well
    
    return processedContent
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/\n/g, '<br>');
  }
  
  // Only do full markdown parsing when streaming is complete
  try {
    // Configure marked with proper options
    marked.setOptions({
      breaks: true,           // Convert \n to <br>
      gfm: true,             // GitHub Flavored Markdown
      headerIds: false,       // Don't add IDs to headers
      mangle: false,         // Don't mangle email addresses
      pedantic: false,       // Use modern defaults
      smartLists: true,      // Use smarter list behavior
      smartypants: false     // Don't use typographic quotes
    });
    
    // Preprocess the text to ensure proper handling of list items
    // Add line breaks before list markers that follow sentences
    const processedText = content
      .replace(/\. \* /g, '.\n\n* ')  // Add line breaks before list items
      .replace(/\. \*\*/g, '.\n\n**'); // Handle bold list items as well
    
    const result = marked.parse(processedText);
    return result;
  } catch (error) {
    console.error('Error parsing markdown:', error);
    // Fallback: try with basic settings
    try {
      return marked.parse(content, { breaks: true });
    } catch (fallbackError) {
      console.error('Fallback markdown parsing also failed:', fallbackError);
      // Last resort: return escaped content with line breaks
      return content.replace(/&/g, '&amp;')
                   .replace(/</g, '&lt;')
                   .replace(/>/g, '&gt;')
                   .replace(/"/g, '&quot;')
                   .replace(/'/g, '&#39;')
                   .replace(/\n/g, '<br>');
    }
  }
};

const MessageRow = ({ m, idx, assets, onEdit, loading }) => {
  // Consume optional action handlers from context via props if passed down in the future
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(m.content || '');

  const save = () => {
    const text = (draft || '').trim();
    if (!text) return setEditing(false);
    setEditing(false);
    onEdit(text);
  };

  return (
    <div
      className={`msg-row ${m.role}`}
      style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', width: '100%', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}
    >
      {/* Left avatar for assistant */}
      {m.role !== 'user' && (
        <img
          src={loading ? assets.net_nodes_animation : assets.network_nodes_logo}
          alt=""
          style={{ width: 28, height: 28 }}
        />
      )}

      {/* Content area */}
      {m.role === 'assistant' ? (
        <div
          className="assistant-bubble"
          style={{ maxWidth: '90%', width: '90%', display: 'flex', flexDirection: 'column', alignItems: 'stretch', gap: 8 }}
        >
          {/* Always use the regular markdown renderer for now to preserve formatting */}
          <span dangerouslySetInnerHTML={{ __html: renderMarkdownContent(m.content, m.isStreaming) }} />
          {m.artifacts && m.artifacts.map((artifact, index) => {
            if (artifact.type === 'plot') {
              try {
                const figure = artifact.content;
                return (
                  <div key={index} style={{ width: '100%', marginTop: '16px' }}>
                    <Plot
                      data={figure.data}
                      layout={figure.layout}
                      config={figure.config}
                      style={{ width: '100%', height: '100%' }}
                      useResizeHandler={true}
                    />
                  </div>
                );
              } catch (e) {
                console.error('Error rendering plot:', e);
                return null;
              }
            }
            return null;
          })}
        </div>
      ) : (
        <div style={{ display: 'flex', maxWidth: '80%', width: '100%', justifyContent: 'flex-end', alignItems: 'flex-end', gap: 8 }}>
          {/* Edit icon outside, left of bubble, bottom-aligned */}
          {!editing && (
            <i
              className="fa-regular fa-pen-to-square action-icon"
              alt="Edit"
              title="Edit message"
              onClick={() => { setDraft(m.content || ''); setEditing(true); }}
            />
          )}
          {/* User bubble */}
          <div className={editing ? "user-bubble full-width" : "user-bubble"} style={{ width: 'auto', maxWidth: '100%' }}>
            {editing ? (
              <div className="full-width" style={{ width: '100%' }}>
                <div className="input-box full-width" style={{ width: '100%', borderRadius: 16 }}>
                  <textarea
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    rows={Math.min(8, Math.max(2, draft.split('\n').length))}
                    style={{ flex: 1, width: '100%', maxHeight: 120 }}
                  />
                  <div style={{ display: 'none' }} />
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'flex-end' }}>
                  <img
                    src={assets.check_icon}
                    alt="Accept"
                    title="Accept and regenerate"
                    onClick={save}
                    style={{ width: 20, height: 20, cursor: 'pointer', opacity: 0.9 }}
                  />
                  <img
                    src={assets.x_icon}
                    alt="Cancel"
                    title="Cancel"
                    onClick={() => setEditing(false)}
                    style={{ width: 20, height: 20, cursor: 'pointer', opacity: 0.9 }}
                  />
                </div>
              </div>
            ) : (
              <div dangerouslySetInnerHTML={{ __html: renderMarkdownContent(m.content) }} />
            )}
          </div>
        </div>
      )}

      {/* Right avatar for user */}
    </div>
  );
};

export default MessageRow;