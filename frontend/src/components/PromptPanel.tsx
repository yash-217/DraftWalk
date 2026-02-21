import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';

export function PromptPanel() {
    const [input, setInput] = useState('');
    const { promptHistory, addPromptMessage, clearPromptHistory, setLoading, patchScene, isLoading, isChatOpen } = useAppStore();
    const historyRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        if (historyRef.current) {
            historyRef.current.scrollTop = historyRef.current.scrollHeight;
        }
    }, [promptHistory]);

    const handleSend = async () => {
        const text = input.trim();
        if (!text) return;

        addPromptMessage({ role: 'user', content: text, timestamp: Date.now() });
        setInput('');
        setLoading(true);

        try {
            const scene = useAppStore.getState().scene;
            const res = await fetch('/api/prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, scene }),
            });
            const data = await res.json();
            if (data.patch) patchScene(data.patch);
            addPromptMessage({ role: 'assistant', content: data.message, timestamp: Date.now() });
        } catch {
            addPromptMessage({ role: 'assistant', content: 'Failed to reach AI service.', timestamp: Date.now() });
        } finally {
            setLoading(false);
        }
    };

    if (!isChatOpen) return null;

    return (
        <aside className="chat-sidebar">
            <div className="chat-sidebar__header">
                <span className="chat-sidebar__title">💬 AI Assistant</span>
                <button
                    className="btn btn--sm"
                    onClick={clearPromptHistory}
                    title="Clear chat history"
                    disabled={promptHistory.length === 0}
                >
                    Clear
                </button>
            </div>

            <div className="chat-sidebar__history" ref={historyRef}>
                {promptHistory.length === 0 && (
                    <div className="chat-sidebar__empty">
                        <div className="chat-sidebar__empty-icon">🤖</div>
                        <p>Ask me to modify your scene.</p>
                        <p className="chat-sidebar__empty-hint">
                            Try "make objects blue", "add a sofa", or "remove object"
                        </p>
                    </div>
                )}
                {promptHistory.map((msg, i) => (
                    <div key={i} className={`chat-sidebar__msg chat-sidebar__msg--${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
            </div>

            <div className="chat-sidebar__input-row">
                <input
                    className="chat-sidebar__input"
                    placeholder="Describe a change…"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) handleSend(); }}
                    disabled={isLoading}
                />
                <button
                    className="chat-sidebar__send"
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                >
                    ➤
                </button>
            </div>
        </aside>
    );
}
