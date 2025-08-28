import React, { useEffect, useState } from 'react';
import { marked } from 'marked';
import './StreamingMarkdownRenderer.css';

/**
 * StreamingMarkdownRenderer handles the progressive rendering of markdown content
 * during streaming, showing safe HTML and pending text appropriately.
 */
const StreamingMarkdownRenderer = ({ 
  content = '', 
  pendingText = '', 
  isComplete = false, 
  isStreaming = false,
  onRenderUpdate 
}) => {
  const [displayContent, setDisplayContent] = useState('');

  useEffect(() => {
    // For streaming, let's just parse the entire raw content as markdown
    // This is simpler and preserves formatting better than complex safe/pending logic
    let finalContent = '';
    
    if (content || pendingText) {
      // Combine all available content
      const fullText = (content || '') + (pendingText || '');
      
      // Try to parse as markdown
      try {
        // Configure marked with proper options
        marked.setOptions({
          breaks: true,
          gfm: true,
          headerIds: false,
          mangle: false,
          pedantic: false,
          smartLists: true,
          smartypants: false
        });
        
        // Preprocess the text to ensure proper handling of list items
        // Add line breaks before list markers that follow sentences
        const processedText = fullText
          .replace(/\. \* /g, '.\n\n* ')  // Add line breaks before list items
          .replace(/\. \*\*/g, '.\n\n**'); // Handle bold list items as well
        
        finalContent = marked.parse(processedText);
      } catch (error) {
        console.error('Error parsing streaming markdown:', error);
        // Fallback: preserve line breaks at minimum
        finalContent = fullText
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;')
          .replace(/\n\n/g, '</p><p>')
          .replace(/\n/g, '<br>');
        
        if (!finalContent.startsWith('<p>')) {
          finalContent = '<p>' + finalContent + '</p>';
        }
      }
    }
    
    setDisplayContent(finalContent);
    
    // Notify parent of render update if callback provided
    if (onRenderUpdate) {
      onRenderUpdate(finalContent);
    }
  }, [content, pendingText, isComplete, onRenderUpdate]);

  // Add cursor effect for streaming
  const cursorStyle = isStreaming && !isComplete ? (
    <span className="streaming-cursor">▊</span>
  ) : null;

  return (
    <div className="streaming-markdown-renderer">
      <span 
        dangerouslySetInnerHTML={{ __html: displayContent }}
      />
      {cursorStyle}
    </div>
  );
};

export default StreamingMarkdownRenderer;