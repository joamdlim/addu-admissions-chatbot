# ADDU Admissions Chatbot - Frontend Implementation

## Overview

This is the frontend implementation of the guided conversation system for the ADDU Admissions Chatbot. It provides a user-friendly interface that guides users through topic selection and maintains context throughout the conversation.

## Features

### 🎯 Guided Conversation Flow

- **Topic Selection**: Visual topic cards for easy selection
- **Contextual Input**: Dynamic input field that shows current topic
- **Action Buttons**: Follow-up actions like "Ask Another Question" or "Change Topic"
- **Auto Topic Detection**: Automatically detects and selects topics from user messages

### 🔄 Dual Mode Support

- **Guided Mode**: Topic-based guided conversation (default)
- **Free Chat Mode**: Traditional open-ended chat
- **Easy Toggle**: Switch between modes with header button

### 🎨 Enhanced UI/UX

- **Responsive Design**: Works on desktop and mobile
- **Visual Feedback**: Loading states, topic indicators, source citations
- **Accessible**: Proper contrast, keyboard navigation, screen reader support
- **Modern Styling**: Clean, professional design matching ADDU branding

## Components

### 1. GuidedChatPage (`/src/pages/GuidedChatPage.jsx`)

Main page component that orchestrates the guided conversation flow.

**Key Features:**

- Manages conversation state (topic_selection, topic_conversation, follow_up)
- Handles API calls to guided chat endpoints
- Displays messages with source citations
- Shows auto-detected topics

**State Management:**

```javascript
const [conversationState, setConversationState] = useState("topic_selection");
const [currentTopic, setCurrentTopic] = useState(null);
const [buttons, setButtons] = useState([]);
const [inputEnabled, setInputEnabled] = useState(false);
```

### 2. GuidedPromptArea (`/src/components/GuidedPromptArea.jsx`)

Dynamic input area that changes based on conversation state.

**Modes:**

- **Topic Selection**: Shows TopicSelector component
- **Text Input**: Shows input field when enabled
- **Action Buttons**: Shows follow-up action buttons

### 3. TopicSelector (`/src/components/TopicSelector.jsx`)

Visual topic selection interface.

**Features:**

- Grid layout for topic cards
- Hover effects and accessibility
- Topic descriptions
- Responsive design

### 4. ActionButtons (`/src/components/ActionButtons.jsx`)

Follow-up action buttons for conversation flow.

**Button Types:**

- **Action buttons**: "Ask Another Question", "Change Topic"
- **Topic buttons**: For topic selection
- **Current topic indicator**: Shows active topic

## API Integration

### Endpoints Used

1. **GET /chatbot/topics/**

   - Loads available topics on page load
   - Used to populate TopicSelector

2. **POST /chatbot/chat/guided/**
   - Main guided conversation endpoint
   - Handles all interaction types (topic_selection, message, action)

### Request Format

```javascript
{
  "user_input": "What are the admission requirements?",
  "action_type": "message", // "message", "topic_selection", "action"
  "action_data": null,      // topic_id or action_id
  "session_id": "session_123"
}
```

### Response Handling

```javascript
{
  "response": "Here are the admission requirements...",
  "state": "follow_up",
  "buttons": [...],
  "input_enabled": false,
  "current_topic": "admissions_enrollment",
  "sources": [...],
  "auto_detected_topic": "Admissions & Enrollment"
}
```

## Conversation Flow

### 1. Initial Load

```
User visits page → Load topics → Show welcome message → Display topic selection
```

### 2. Topic Selection

```
User clicks topic → Send topic_selection request → Show welcome for topic → Enable input
```

### 3. Question & Answer

```
User types question → Send message request → Show answer → Display action buttons
```

### 4. Follow-up Actions

```
User clicks "Ask Another" → Enable input in same topic
User clicks "Change Topic" → Return to topic selection
```

### 5. Auto Topic Detection

```
User types without selecting topic → Auto-detect topic → Process question → Show answer
```

## Styling & Design

### Color Scheme

- **Primary**: `#063970` (ADDU Blue)
- **Secondary**: `#052a5a` (Darker Blue)
- **Background**: `#f9fafb` (Light Gray)
- **Text**: `#111827` (Dark Gray)

### Component Styling

- **Topic Cards**: White background, hover effects, rounded corners
- **Messages**: User (blue), Bot (white with border), Error (red tint)
- **Buttons**: Primary (blue), Secondary (outlined), Disabled (gray)
- **Input**: Clean design with topic context indicator

### Responsive Design

- **Desktop**: 900px max width, grid layout for topics
- **Mobile**: Single column, stacked layout, touch-friendly buttons

## Usage Examples

### Basic Usage

```jsx
import { GuidedChatPage } from "./pages/GuidedChatPage";

function App() {
  return <GuidedChatPage />;
}
```

### With Mode Toggle

```jsx
const [isGuidedMode, setIsGuidedMode] = useState(true);

return (
  <div>
    <button onClick={() => setIsGuidedMode(!isGuidedMode)}>
      {isGuidedMode ? "Guided" : "Free Chat"}
    </button>
    {isGuidedMode ? <GuidedChatPage /> : <ChatPage />}
  </div>
);
```

## Configuration

### API Base URL

Update the base URL in components:

```javascript
const BASE_URL = "http://127.0.0.1:8000/chatbot";
```

### Styling Customization

Modify Tailwind classes in components:

```javascript
className = "bg-[#063970] text-white"; // ADDU Blue
```

### Topic Display

Topics are loaded dynamically from the backend. No frontend configuration needed.

## Development

### Prerequisites

- Node.js 16+
- React 18+
- Tailwind CSS
- React Router DOM
- React Markdown

### Installation

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

## Testing

### Manual Testing Checklist

1. **Topic Selection**

   - [ ] Topics load on page load
   - [ ] Topic cards are clickable
   - [ ] Topic selection shows welcome message
   - [ ] Input is enabled after topic selection

2. **Conversation Flow**

   - [ ] Questions are processed correctly
   - [ ] Responses show with proper formatting
   - [ ] Sources are displayed when available
   - [ ] Action buttons appear after responses

3. **Follow-up Actions**

   - [ ] "Ask Another Question" enables input
   - [ ] "Change Topic" returns to topic selection
   - [ ] Current topic is displayed correctly

4. **Auto Topic Detection**

   - [ ] Typing without topic selection works
   - [ ] Auto-detected topic is shown
   - [ ] Questions are processed in detected topic

5. **Mode Toggle**

   - [ ] Toggle between Guided and Free Chat works
   - [ ] State is preserved when switching modes
   - [ ] UI updates correctly for each mode

6. **Error Handling**

   - [ ] Network errors are handled gracefully
   - [ ] Invalid responses show error messages
   - [ ] Loading states work correctly

7. **Responsive Design**
   - [ ] Works on desktop (1200px+)
   - [ ] Works on tablet (768px-1200px)
   - [ ] Works on mobile (320px-768px)

### Browser Testing

- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari
- [ ] Edge

## Troubleshooting

### Common Issues

1. **Topics not loading**

   - Check backend is running on port 8000
   - Verify `/chatbot/topics/` endpoint is accessible
   - Check browser console for CORS errors

2. **Messages not sending**

   - Verify `/chatbot/chat/guided/` endpoint is working
   - Check request payload format
   - Look for network errors in browser dev tools

3. **Styling issues**

   - Ensure Tailwind CSS is properly configured
   - Check for conflicting CSS rules
   - Verify responsive breakpoints

4. **State management issues**
   - Check React dev tools for state updates
   - Verify API responses are updating state correctly
   - Look for console errors in state setters

### Debug Mode

Enable debug logging:

```javascript
console.log("🚀 Guided request:", requestBody);
console.log("📨 Guided response:", data);
```

### Performance Optimization

1. **Lazy Loading**: Components are loaded on demand
2. **Memoization**: Use React.memo for expensive components
3. **Debouncing**: Add input debouncing for better UX
4. **Caching**: Cache topics and session data

## Future Enhancements

1. **Voice Interface**: Add speech-to-text and text-to-speech
2. **Multi-language**: Support for Filipino and other languages
3. **Offline Mode**: Cache responses for offline access
4. **Analytics**: Track user interactions and popular topics
5. **Accessibility**: Enhanced screen reader support
6. **Mobile App**: React Native version
7. **Chatbot Avatar**: Animated character for better engagement
8. **Rich Media**: Support for images, videos, and documents

## Contributing

1. Follow React best practices
2. Use TypeScript for new components
3. Add proper PropTypes or TypeScript interfaces
4. Include unit tests for new features
5. Follow the existing code style
6. Update documentation for changes

## License

This project is part of the ADDU Admissions Chatbot system. All rights reserved.
