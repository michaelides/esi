/**
 * MarkdownStateDetector provides utilities for detecting markdown structures
 * and determining safe parsing boundaries during streaming.
 */
export class MarkdownStateDetector {
  
  /**
   * Detect if content is safe to render based on markdown state
   * @param {string} content - Content to analyze
   * @returns {boolean} - Whether content is safe to parse
   */
  static isSafeToRender(content) {
    const state = this.detectState(content);
    return !state.hasIncompleteStructures;
  }

  /**
   * Find the last safe boundary where markdown can be safely parsed
   * @param {string} content - Content to analyze
   * @returns {number} - Index of safe boundary
   */
  static findSafeBoundary(content) {
    let boundary = 0;
    let inCodeBlock = false;
    let inInlineCode = false;
    let inBold = false;
    let inItalic = false;
    
    for (let i = 0; i < content.length; i++) {
      const char = content[i];
      const nextChar = content[i + 1];
      const prevChar = content[i - 1];

      // Handle code blocks
      if (char === '`' && nextChar === '`' && content[i + 2] === '`') {
        if (!inCodeBlock) {
          inCodeBlock = true;
          i += 2;
        } else {
          inCodeBlock = false;
          boundary = i + 3;
          i += 2;
        }
        continue;
      }

      if (inCodeBlock) continue;

      // Handle inline code
      if (char === '`' && prevChar !== '\\') {
        inInlineCode = !inInlineCode;
        if (!inInlineCode) boundary = i + 1;
        continue;
      }

      if (inInlineCode) continue;

      // Handle bold/italic
      if (char === '*' && prevChar !== '\\') {
        if (nextChar === '*') {
          inBold = !inBold;
          if (!inBold) boundary = i + 2;
          i++;
        } else {
          inItalic = !inItalic;
          if (!inItalic) boundary = i + 1;
        }
        continue;
      }

      // Line boundaries are safe if not in incomplete structures
      if (char === '\n' && !inCodeBlock && !inInlineCode && !inBold && !inItalic) {
        boundary = i + 1;
      }
    }

    // If no incomplete structures at end, entire content is safe
    if (!inCodeBlock && !inInlineCode && !inBold && !inItalic) {
      boundary = content.length;
    }

    return boundary;
  }

  /**
   * Comprehensive state detection
   * @param {string} content - Content to analyze
   * @returns {object} - Markdown state summary
   */
  static detectState(content) {
    const codeBlocks = this.detectCodeBlocks(content);
    const inlineCode = this.detectInlineCode(content);
    const formatting = this.detectInlineFormatting(content);
    
    return {
      codeBlocks,
      inlineCode,
      formatting,
      hasIncompleteStructures: (
        codeBlocks.hasIncompleteBlock ||
        inlineCode.hasIncompleteCode ||
        formatting.bold.incomplete ||
        formatting.italic.incomplete
      )
    };
  }

  /**
   * Detect code block state
   */
  static detectCodeBlocks(content) {
    const lines = content.split('\n');
    let inCodeBlock = false;
    
    for (const line of lines) {
      if (line.match(/^`{3,}/)) {
        inCodeBlock = !inCodeBlock;
      }
    }
    
    return { hasIncompleteBlock: inCodeBlock };
  }

  /**
   * Detect inline code state
   */
  static detectInlineCode(content) {
    let inInlineCode = false;
    
    for (let i = 0; i < content.length; i++) {
      if (content[i] === '`' && content[i - 1] !== '\\') {
        inInlineCode = !inInlineCode;
      }
    }
    
    return { hasIncompleteCode: inInlineCode };
  }

  /**
   * Detect inline formatting state
   */
  static detectInlineFormatting(content) {
    let boldCount = 0;
    let italicCount = 0;
    
    let i = 0;
    while (i < content.length) {
      if (content[i] === '*' && content[i - 1] !== '\\') {
        if (content[i + 1] === '*') {
          boldCount++;
          i += 2;
        } else {
          italicCount++;
          i++;
        }
      } else {
        i++;
      }
    }
    
    return {
      bold: { incomplete: boldCount % 2 !== 0 },
      italic: { incomplete: italicCount % 2 !== 0 }
    };
  }
}

export default MarkdownStateDetector;