import { marked } from 'marked';

/**
 * MarkdownBufferManager handles safe markdown parsing during streaming.
 * It maintains both raw and safe buffers to prevent incomplete markdown from being rendered incorrectly.
 */
export class MarkdownBufferManager {
  constructor() {
    this.rawBuffer = '';
    this.safeBuffer = '';
    this.pendingMarkdown = '';
    this.inCodeBlock = false;
    this.codeBlockFence = '';
    this.inInlineCode = false;
    this.inBoldItalic = false;
    this.inLink = false;
    this.listDepth = 0;
    this.inTable = false;
    this.lastSafeBoundary = 0;
    
    // Configure marked options
    this.markedOptions = {
      breaks: true,
      gfm: true,
      tables: true,
      sanitize: false,
      headerIds: false,
      mangle: false,
      pedantic: false,
      smartLists: true,
      smartypants: false
    };
    
    // Configure marked with proper options
    marked.setOptions(this.markedOptions);
  }

  /**
   * Add a new token to the buffer and return safe content for rendering
   * @param {string} token - The new token to add
   * @returns {object} - {content: string, isComplete: boolean, isStreaming: boolean}
   */
  addToken(token) {
    this.rawBuffer += token;
    this.updateSafeBuffer();
    return this.getSafeContent();
  }

  /**
   * Update the safe buffer by analyzing markdown state
   */
  updateSafeBuffer() {
    const content = this.rawBuffer;
    let safeBoundary = this.findSafeBoundary(content);
    
    // Extract safe content up to the boundary
    this.safeBuffer = content.substring(0, safeBoundary);
    this.pendingMarkdown = content.substring(safeBoundary);
    this.lastSafeBoundary = safeBoundary;
  }

  /**
   * Find the safe boundary where we can safely parse markdown
   * @param {string} content - Content to analyze
   * @returns {number} - Index of safe boundary
   */
  findSafeBoundary(content) {
    let boundary = 0;
    let inCodeBlock = false;
    let codeBlockFence = '';
    let inInlineCode = false;
    let inBold = false;
    let inItalic = false;
    let inLink = false;
    let linkBracketCount = 0;
    let i = 0;

    while (i < content.length) {
      const char = content[i];
      const nextChar = content[i + 1];
      const prevChar = content[i - 1];

      // Handle code blocks (``` or ~~~)
      if (!inInlineCode && (char === '`' || char === '~')) {
        const fenceMatch = content.substring(i).match(/^(`{3,}|~{3,})/);
        if (fenceMatch) {
          const fence = fenceMatch[1];
          if (!inCodeBlock) {
            // Starting a code block
            inCodeBlock = true;
            codeBlockFence = fence;
            i += fence.length;
            continue;
          } else if (fence.startsWith(codeBlockFence[0]) && fence.length >= codeBlockFence.length) {
            // Ending a code block
            inCodeBlock = false;
            codeBlockFence = '';
            i += fence.length;
            // Update boundary after closing code block
            boundary = i;
            continue;
          }
        }
      }

      // If we're in a code block, skip all other markdown parsing
      if (inCodeBlock) {
        i++;
        continue;
      }

      // Handle inline code
      if (char === '`' && prevChar !== '\\') {
        if (!inInlineCode) {
          inInlineCode = true;
        } else {
          inInlineCode = false;
          // Update boundary after closing inline code
          boundary = i + 1;
        }
        i++;
        continue;
      }

      // If we're in inline code, skip other markdown parsing
      if (inInlineCode) {
        i++;
        continue;
      }

      // Handle bold/italic
      if (char === '*' && prevChar !== '\\') {
        if (nextChar === '*') {
          // Bold
          if (!inBold) {
            inBold = true;
          } else {
            inBold = false;
            boundary = i + 2;
          }
          i += 2;
          continue;
        } else {
          // Italic
          if (!inItalic) {
            inItalic = true;
          } else {
            inItalic = false;
            boundary = i + 1;
          }
          i++;
          continue;
        }
      }

      // Handle links [text](url)
      if (char === '[' && prevChar !== '\\') {
        inLink = true;
        linkBracketCount = 1;
      } else if (inLink && char === '[') {
        linkBracketCount++;
      } else if (inLink && char === ']') {
        linkBracketCount--;
        if (linkBracketCount === 0) {
          // Check if followed by (url)
          const urlMatch = content.substring(i + 1).match(/^\([^)]*\)/);
          if (urlMatch) {
            inLink = false;
            i += urlMatch[0].length;
            boundary = i + 1;
          }
        }
      }

      // Handle line boundaries (safe points for lists, headers, etc.)
      if (char === '\n') {
        // If we're not in any incomplete markdown structure, this is a safe boundary
        if (!inCodeBlock && !inInlineCode && !inBold && !inItalic && !inLink) {
          // Check for paragraph breaks (double newlines) - these are always safe
          if (nextChar === '\n' || (i === content.length - 1)) {
            boundary = i + 1;
          } else {
            // Single newline - check if next line starts with safe content
            const nextLineMatch = content.substring(i + 1).match(/^(#{1,6}\s|\d+\.\s|[-*+]\s|\s*$|[^\n]*$)/);
            if (nextLineMatch) {
              boundary = i + 1;
            }
          }
        }
      }

      i++;
    }

    // If we're not in any incomplete structure and at end, entire content is safe
    if (!inCodeBlock && !inInlineCode && !inBold && !inItalic && !inLink) {
      boundary = content.length;
    }

    return boundary;
  }

  /**
   * Get safe content for rendering
   * @returns {object} - Render information
   */
  getSafeContent() {
    const hasUnsafeContent = this.pendingMarkdown.length > 0;
    
    try {
      // Parse the safe content
      let safeHtml = this.safeBuffer ? marked.parse(this.safeBuffer, this.markedOptions) : '';
      
      // If we have pending content, try to parse safe portions of it
      let pendingText = '';
      if (hasUnsafeContent) {
        const pending = this.pendingMarkdown;
        
        // Look for complete lines that can be safely parsed
        const lines = pending.split('\n');
        let safePendingLines = [];
        let unsafePendingLines = [];
        
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          
          // Check if this line contains complete markdown elements
          const isCompleteLine = (
            /^#{1,6}\s.*$/.test(line) ||           // Headers
            /^\s*[-*+]\s.*$/.test(line) ||        // List items
            /^\s*\d+\.\s.*$/.test(line) ||       // Numbered lists
            /^\s*>\s.*$/.test(line) ||           // Blockquotes
            /^\s*$/.test(line) ||                // Empty lines
            (line.length > 0 && !line.includes('*') && !line.includes('`') && !line.includes('[')) // Plain text lines
          );
          
          if (isCompleteLine || i < lines.length - 1) {
            // Complete line or not the last line
            safePendingLines.push(line);
          } else {
            // Last incomplete line
            unsafePendingLines.push(line);
          }
        }
        
        // Parse safe pending lines
        if (safePendingLines.length > 0) {
          const safePendingText = safePendingLines.join('\n');
          try {
            // Preprocess the text to ensure proper handling of list items
            // Add line breaks before list markers that follow sentences
            const processedText = safePendingText
              .replace(/\. \* /g, '.\n\n* ')  // Add line breaks before list items
              .replace(/\. \*\*/g, '.\n\n**'); // Handle bold list items as well
            
            const safePendingHtml = marked.parse(processedText, this.markedOptions);
            safeHtml += safePendingHtml;
          } catch (error) {
            // If parsing fails, treat as unsafe
            unsafePendingLines = [...safePendingLines, ...unsafePendingLines];
          }
        }
        
        // Keep unsafe lines as pending text
        pendingText = unsafePendingLines.join('\n');
      }
      
      return {
        content: safeHtml,
        pendingText: pendingText,
        isComplete: !hasUnsafeContent || pendingText.length === 0,
        isStreaming: true,
        rawContent: this.rawBuffer
      };
    } catch (error) {
      console.error('Error parsing markdown:', error);
      // Fallback to raw text on parsing error
      return {
        content: '',
        pendingText: this.rawBuffer,
        isComplete: false,
        isStreaming: true,
        rawContent: this.rawBuffer
      };
    }
  }

  /**
   * Finalize the buffer and return complete parsed content
   * @returns {string} - Final HTML content
   */
  finalize() {
    try {
      // Preprocess the text to ensure proper handling of list items
      // Add line breaks before list markers that follow sentences
      const processedText = this.rawBuffer
        .replace(/\. \* /g, '.\n\n* ')  // Add line breaks before list items
        .replace(/\. \*\*/g, '.\n\n**'); // Handle bold list items as well
      
      const finalHtml = marked.parse(processedText, this.markedOptions);
      this.reset();
      return finalHtml;
    } catch (error) {
      console.error('Error in final markdown parsing:', error);
      const result = this.rawBuffer;
      this.reset();
      return result;
    }
  }

  /**
   * Reset the buffer manager for a new message
   */
  reset() {
    this.rawBuffer = '';
    this.safeBuffer = '';
    this.pendingMarkdown = '';
    this.inCodeBlock = false;
    this.codeBlockFence = '';
    this.inInlineCode = false;
    this.inBoldItalic = false;
    this.inLink = false;
    this.listDepth = 0;
    this.inTable = false;
    this.lastSafeBoundary = 0;
  }

  /**
   * Get current buffer state for debugging
   * @returns {object} - Current state
   */
  getState() {
    return {
      rawBufferLength: this.rawBuffer.length,
      safeBufferLength: this.safeBuffer.length,
      pendingLength: this.pendingMarkdown.length,
      inCodeBlock: this.inCodeBlock,
      inInlineCode: this.inInlineCode,
      lastSafeBoundary: this.lastSafeBoundary
    };
  }
}

export default MarkdownBufferManager;