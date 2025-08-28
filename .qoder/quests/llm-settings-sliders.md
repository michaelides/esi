# LLM Settings Sliders Implementation Design

## Overview

This document outlines the design for implementing two new settings sliders in the sidebar:
1. **Creativity Slider** - Controls the LLM temperature parameter with model-specific constraints
2. **Verbosity Slider** - Controls the response length with 5 discrete levels (1=Laconic to 5=Very Verbose)

The implementation will follow the existing pattern established by the ModelSelector and ThemeSelector components, integrating directly into the sidebar settings section.

## Architecture

### Component Structure
```
src/
├── components/
│   ├── Sidebar/
│   │   ├── Sidebar.jsx (existing)
│   │   └── Sidebar.css (existing)
│   └── Settings/
│       ├── TemperatureSlider.jsx (new)
│       ├── VerbositySlider.jsx (new)
│       ├── TemperatureSlider.css (new)
│       └── VerbositySlider.css (new)
└── context/
    └── Context.jsx (existing, with modifications)
```

### State Management

The settings will be managed through the existing Context API with the following state variables:

1. `temperature` (Number) - Current temperature value (0.0 to model-specific max)
2. `verbosity` (Integer) - Current verbosity level (1-5)
3. `setTemperature` (Function) - Updates temperature with model constraints
4. `setVerbosity` (Function) - Updates verbosity level

### Data Flow
```
[User Interaction] → [Slider Component] → [Context State] → [API Request]
```

When a user adjusts the sliders:
1. The slider component calls the appropriate setter function from context
2. The context updates the state and persists to localStorage
3. When making API requests, the values are included in the options payload

## Component Design

### TemperatureSlider Component

#### Features
- Label: "Creativity"
- Slider range: 0.0 to model-specific maximum (1.0 for Gemini/Mistral, 2.0 for others)
- Real-time constraint adjustment when model changes
- Tooltip on hover explaining the parameter
- Visual feedback for current value

### VerbositySlider Component

#### Features
- Label: "Verbosity"
- Slider range: 1-5 with discrete steps
- Labels for each level: 1=Laconic, 2=Concise, 3=Moderate, 4=Verbose, 5=Very Verbose
- Visual feedback for current selection

## Styling

The sliders will follow the existing styling patterns established by the ModelSelector component with the following specifications:

1. Each slider will have:
   - Label on a separate line above the slider
   - Slider control spanning the full width of the sidebar
   - Visual feedback for current value/selection

2. TemperatureSlider specific styling:
   - Tooltip that appears on hover explaining the temperature parameter
   - Purple accent color (#a855f7) for the slider thumb to match existing UI
   - 4px track height with rounded corners

3. VerbositySlider specific styling:
   - Display of current verbosity level description (Laconic to Very Verbose)
   - Consistent styling with other sidebar controls
   - Purple accent color (#a855f7) for the slider thumb

4. Both sliders will:
   - Use 16px circular thumb that scales on hover
   - Have smooth transitions for interactive elements
   - Support dark mode with appropriate color adjustments

## Integration with Existing Code

### Sidebar Integration

The new slider components will be integrated into the existing settings section in the sidebar, following the same pattern as the ModelSelector and ThemeSelector components.

### Context Integration

The context already has the required state variables and setters, but we need to ensure the temperature constraint logic works properly with the new slider component.

## API Integration

The temperature and verbosity values will be passed to the backend API in the options payload alongside other settings like the selected model.

## Model-Specific Constraints

The temperature slider will automatically adjust its maximum value based on the selected model:

- **Gemini models**: Max temperature = 1.0
- **Mistral models**: Max temperature = 1.0
- **Other models**: Max temperature = 2.0

When a user changes models, the temperature will be automatically constrained to the new model's maximum if it exceeds it.

## User Experience

### Slider Layout
Each slider will be implemented with:
1. Label on a separate line above the slider
2. Slider control spanning the full width of the sidebar
3. Visual feedback for current value/selection

### Tooltip Implementation
The Creativity slider will have a hover tooltip that explains:
- What the temperature parameter controls
- Effect of higher values (more creative/unpredictable)
- Effect of lower values (more consistent/deterministic)

### Responsiveness
The sliders will:
- Span the full width of the sidebar when extended
- Be hidden when the sidebar is collapsed
- Maintain consistent styling with dark/light mode

## Testing

### Unit Tests
1. Test temperature constraint logic with different models
2. Test localStorage persistence for both settings
3. Test slider components with various input values
4. Test tooltip visibility on hover

### Integration Tests
1. Test that slider values are correctly passed to API
2. Test model change triggers temperature constraint
3. Test that settings persist across sessions

## Security Considerations

- All settings are stored in localStorage and not transmitted to external servers
- Temperature values are validated and constrained before being passed to LLM APIs
- No sensitive information is exposed through the slider components